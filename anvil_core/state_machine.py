"""The 10-step state machine (§1, §7).

One linear pass, top to bottom, per generation request. Every step checkpoints to
session.json when it completes, which is what makes resume able to skip work that
already happened instead of redoing it.

    1. mode_select          which of the 3 input modes
    2. style_resolve        preset lookup, or custom captioning
    3. checklist_fill       mandatory fields must be complete (Mode 1 only)
    4. build_prompt         merge checklist + style into the final prompt
    5. image_generate       Text→Image mode only
    6. determine_3d_input   resolve what step 7 generates from
    7. model_generate       image-or-text → raw mesh (or in-scene bpy geometry)
    8. postprocess          style-driven mesh cleanup
    9. import_to_blender    bpy import into the current scene
   10. edit_loop            handed off to the UI; not run inline

edit_loop is step 10 but is NOT executed here — it's interactive, driven by the
UI. run_generation_pipeline() completes steps 1-9 and returns; the server then
serves edit requests against the finished session. This is the §7 fix that stops
edits recursing back through the top-level entry point.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from . import (
    blender_bridge,
    events,
    genrequest,
    image_gen,
    llm,
    model_gen,
    postprocess,
    procedural_gen,
    sessions,
    view_preprocess,
    view_validate,
)
from .config import SETTINGS
from .gpu_check import check_gpu_capability
from .schemas import Session, missing_mandatory
from .status import PIPELINE_STATUS
from .styles import get_style


class PipelineError(RuntimeError):
    pass


class ChecklistIncomplete(PipelineError):
    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Mandatory fields still empty: {', '.join(missing)}")


def _step_paths(session: Session) -> dict[str, Path]:
    d = sessions.session_dir(session.session_id)
    return {
        "dir": d,
        "concept_image": d / "concept.png",
        "raw_mesh": d / "raw_mesh.glb",
        "final_mesh": d / "final_mesh.glb",
        "request": d / "request.json",
        "views": d / "views",
    }


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def run_anvil_session(session_id: str | None = None, **initial) -> Session:
    """Load or create the session, verify scene state if resuming, then run the
    generation pipeline once. Does NOT call edit_loop — the UI drives that."""
    session = sessions.load_or_create_session(session_id, **initial) if initial or session_id \
        else sessions.load_or_create_session()

    _verify_procedural_scene_on_resume(session)
    return run_generation_pipeline(session)


def _verify_procedural_scene_on_resume(session: Session) -> None:
    """§7/§9 — a procedural session resumed against a diverged scene must roll
    step 7 back rather than reporting success with nothing in the scene.

    Only relevant at load time, not inside run_generation_pipeline (which also
    runs for in-session shape-edit regeneration, where this check doesn't apply).
    """
    if not session.final_mesh_path or not session.final_mesh_path.startswith("in_scene:"):
        return

    object_name = session.final_mesh_path.split(":", 1)[1]
    if not blender_bridge.blender_available():
        return

    result = blender_bridge.verify_scene_object(
        object_name, session.scene_object_uuid or "", session_id=session.session_id
    )
    diverged = (
        not result.get("ok")
        or not result.get("present")
        or (session.blend_file_path and result.get("blend_file") and result["blend_file"] != session.blend_file_path)
    )
    if diverged:
        events.log(
            session.session_id,
            "Previous session's object isn't in the current scene — rolling step 7 back for regeneration.",
            "warn",
        )
        session.final_mesh_path = None
        session.scene_object_uuid = None
        for step in ("model_generate", "postprocess", "import_to_blender"):
            if step in session.completed_steps:
                session.completed_steps.remove(step)
        session.current_step = min(session.current_step, 6)
        sessions.save_session(session)


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def run_generation_pipeline(
    session: Session, *, force_from: int = 0, stop_after: str | None = None
) -> Session:
    """Steps 1-9. Idempotent per step: an already-completed step is skipped
    unless force_from says to redo from a given index (shape edits use this).

    `stop_after` parks the run after a named step instead of carrying on, leaving
    the session `awaiting_review`. That is what lets Text→Image show its concept
    image and wait: 3D generation is the expensive half, and spending three
    minutes on a picture the user would have rejected in one glance is waste the
    pipeline is well placed to avoid. Resuming is the ordinary resume path —
    every step already checkpoints, so no second code path is needed.
    """
    sid = session.session_id
    total = 9  # steps 1-9; step 10 is the interactive edit loop

    capability = check_gpu_capability()
    if capability["blocked"]:
        raise PipelineError(capability["warning"] or "No usable GPU detected")

    session.status = "running"
    session.error = None
    sessions.save_session(session)

    try:
        paths = _step_paths(session)

        # ---- 1. mode_select ------------------------------------------------
        if _should_run(session, "mode_select", 1, force_from):
            events.step_start(sid, "mode_select", 1, total, "Select input mode")
            if session.mode not in ("text_to_image", "text", "image_upload"):
                raise PipelineError(f"Unknown input mode: {session.mode!r}")
            session.mark_step("mode_select")
            sessions.save_session(session)
            events.step_done(sid, "mode_select", 1, total)

        # ---- 2. style_resolve ----------------------------------------------
        style = get_style(session.style_id)
        if _should_run(session, "style_resolve", 2, force_from):
            events.step_start(sid, "style_resolve", 2, total, "Resolve style preset")
            if session.style_id == "custom" and not session.custom_style_modifier:
                # Custom needs a reference resolved per-generation
                if session.uploaded_image_path:
                    session.custom_style_modifier = llm.caption_style_reference(
                        image_path=session.uploaded_image_path, session_id=sid
                    )
                elif session.checklist.get("style_desc"):
                    session.custom_style_modifier = llm.caption_style_reference(
                        text=session.checklist["style_desc"], session_id=sid
                    )
                else:
                    raise PipelineError(
                        "The Custom style needs a reference — upload an image or fill in the "
                        "style descriptor field."
                    )
                events.log(sid, f"Custom style resolved: {session.custom_style_modifier[:80]}")
            session.mark_step("style_resolve")
            sessions.save_session(session)
            events.step_done(sid, "style_resolve", 2, total)

        # ---- 3. checklist_fill ---------------------------------------------
        if _should_run(session, "checklist_fill", 3, force_from):
            events.step_start(sid, "checklist_fill", 3, total, "Verify checklist")
            missing = missing_mandatory(session.mode, session.checklist)
            if missing:
                # Mode 1's mandatory fields BLOCK generation (§4.1). The UI is
                # responsible for collecting them; we refuse rather than guess.
                raise ChecklistIncomplete(missing)
            session.mark_step("checklist_fill")
            sessions.save_session(session)
            events.step_done(sid, "checklist_fill", 3, total)

        # ---- 4. build_prompt -----------------------------------------------
        if _should_run(session, "build_prompt", 4, force_from):
            events.step_start(sid, "build_prompt", 4, total, "Build prompt")
            if session.prompt_override and session.final_prompt:
                # The user edited this during concept-image review; rebuilding from
                # the checklist would throw their wording away without telling them.
                events.log(sid, "Using the edited prompt as-is")
            else:
                session.final_prompt = build_prompt(session, style)
            events.log(sid, f"Prompt: {session.final_prompt[:160]}")
            session.mark_step("build_prompt")
            sessions.save_session(session)
            events.step_done(sid, "build_prompt", 4, total)

        # ---- 5. image_generate (Text→Image only) ----------------------------
        if session.mode == "text_to_image" and not session.procedural:
            if _should_run(session, "image_generate", 5, force_from):
                events.step_start(sid, "image_generate", 5, total, "Generate concept image")
                session.generated_image_path = str(
                    image_gen.generate_image(session.final_prompt, paths["concept_image"], session_id=sid)
                )
                session.mark_step("image_generate")
                sessions.save_session(session)
                events.step_done(sid, "image_generate", 5, total)
        else:
            session.mark_step("image_generate")   # not applicable in this mode

        if stop_after == "image_generate":
            session.status = "awaiting_review"
            sessions.save_session(session)
            events.emit(
                "session_paused",
                session_id=sid,
                at="image_generate",
                image_path=session.generated_image_path,
                prompt=session.final_prompt,
                label=session.display_label(),
            )
            # The status snapshot has to be closed out too, or the panel shows a
            # run that never ends.
            PIPELINE_STATUS.finish(sid)
            return session

        # ---- 6. determine_3d_input ------------------------------------------
        if _should_run(session, "determine_3d_input", 6, force_from):
            events.step_start(sid, "determine_3d_input", 6, total, "Determine 3D input")
            session.three_d_input_kind = _determine_3d_input(session)
            events.log(sid, f"Step 7 will generate from: {session.three_d_input_kind}")
            _build_gen_request(session, paths)
            session.mark_step("determine_3d_input")
            sessions.save_session(session)
            events.step_done(sid, "determine_3d_input", 6, total)

        # ---- 6b. preprocess + validate views --------------------------------
        # Runs when a GenRequest exists (image-conditioned path). Preprocesses
        # each view through the 5-step pipeline (EXIF, matting, crop, scale,
        # resize) and then validates cross-view consistency. Hard-fail checks
        # block generation; warnings are logged but don't stop the run.
        if _should_run(session, "preprocess_views", 6, force_from):
            request = _load_gen_request(paths)
            if request is not None and request.views:
                events.log(sid, f"Preprocessing {len(request.views)} view(s)")
                try:
                    view_preprocess.preprocess_request(
                        request, session_id=sid, views_dir=paths["views"],
                    )
                except FileNotFoundError as exc:
                    events.log(sid, f"View preprocess skipped: {exc}", "warn")
                else:
                    # Validate the preprocessed views
                    report = view_validate.validate_request(
                        request, session_id=sid, use_dino=False,
                    )
                    for w in report.warnings:
                        events.log(sid, f"View warning: {w.message}", "warn")
                    if not report.valid:
                        fail_msgs = "; ".join(c.message for c in report.failures)
                        raise PipelineError(f"View validation failed: {fail_msgs}")

                    # Re-checkpoint the request with preprocessed scale_ratio values
                    paths["request"].write_text(
                        json.dumps(request.to_dict(), indent=2), encoding="utf-8",
                    )
            session.mark_step("preprocess_views")
            sessions.save_session(session)

        # ---- 7. model_generate ----------------------------------------------
        if _should_run(session, "model_generate", 7, force_from):
            if session.procedural:
                events.step_start(sid, "model_generate", 7, total, "Generate procedural geometry")
                _run_procedural(session, style)
            else:
                events.step_start(sid, "model_generate", 7, total, "Generate 3D mesh")
                image_for_3d = (
                    session.uploaded_image_path if session.mode == "image_upload"
                    else session.generated_image_path
                ) if session.three_d_input_kind == "image" else None

                session.raw_mesh_path = str(model_gen.generate_mesh(
                    prompt=session.final_prompt,
                    image_path=image_for_3d,
                    out_path=paths["raw_mesh"],
                    style=style,
                    flags=session.flags,
                    session_id=sid,
                    request=_load_gen_request(paths),
                ))
            session.mark_step("model_generate")
            sessions.save_session(session)
            events.step_done(sid, "model_generate", 7, total)

        # ---- 8. postprocess --------------------------------------------------
        if _should_run(session, "postprocess", 8, force_from):
            if session.procedural:
                # Procedural geometry is already clean and lives in-scene — there's
                # no mesh file to run a decimate/remesh chain over (§2.6.3).
                events.log(sid, "Procedural mode — postprocess chain skipped (geometry is already in-scene)")
            else:
                events.step_start(sid, "postprocess", 8, total, "Post-process mesh")
                session.final_mesh_path = str(postprocess.run_postprocess(
                    Path(session.raw_mesh_path), paths["final_mesh"], style, session_id=sid
                ))
                events.step_done(sid, "postprocess", 8, total)
            session.mark_step("postprocess")
            sessions.save_session(session)

        # ---- 9. import_to_blender --------------------------------------------
        if _should_run(session, "import_to_blender", 9, force_from):
            if session.procedural:
                events.log(sid, "Procedural mode — object was constructed directly in the scene")
            else:
                events.step_start(sid, "import_to_blender", 9, total, "Import to scene")
                object_name = f"ANVIL_{sid[:6]}"
                session.scene_object_uuid = uuid.uuid4().hex
                result = blender_bridge.import_mesh(
                    Path(session.final_mesh_path), session.scene_object_uuid, object_name, session_id=sid
                )
                if result.get("skipped"):
                    events.log(sid, "Blender not configured — mesh file is on disk but not imported", "warn")
                elif not result.get("ok"):
                    raise PipelineError(result.get("error") or "Import into Blender failed")
                else:
                    session.blend_file_path = str(blender_bridge.blend_file_path())
                events.step_done(sid, "import_to_blender", 9, total)
            session.mark_step("import_to_blender")
            sessions.save_session(session)

        # step 10 (edit_loop) is interactive — the UI owns it from here.
        session.status = "done"
        sessions.save_session(session)

        stats = (
            postprocess.mesh_stats(Path(session.final_mesh_path))
            if session.final_mesh_path and not session.final_mesh_path.startswith("in_scene:")
            else {"tris": 0, "verts": 0}
        )
        events.session_done(
            sid,
            style_id=session.style_id,
            label=session.display_label(),
            mesh_path=session.final_mesh_path,
            image_path=session.generated_image_path,
            procedural=session.procedural,
            stats=stats,
        )
        return session

    except Exception as exc:
        session.status = "failed"
        session.error = f"{type(exc).__name__}: {exc}"
        sessions.save_session(session)
        events.session_failed(sid, str(exc))
        raise


def _should_run(session: Session, step: str, index: int, force_from: int) -> bool:
    if force_from and index >= force_from:
        return True
    return not session.has_completed(step)


# ---------------------------------------------------------------------------
# step implementations that are worth naming
# ---------------------------------------------------------------------------

def build_prompt(session: Session, style) -> str:
    """Step 4 — merge checklist fields + style modifier into one prompt string.

    Mode 3 contributes no prompt text at all: its fields are generation flags,
    not descriptors (§4.3), so the prompt stays empty and step 6 routes the
    uploaded image straight to the backend.
    """
    if session.mode == "image_upload":
        return ""

    ordered = [
        "object_type", "style_desc", "primary_material", "color_palette",
        "lighting", "scale_reference", "wear_and_tear", "symmetry", "background_context",
    ]
    parts = [str(session.checklist[k]).strip() for k in ordered
             if session.checklist.get(k) and str(session.checklist[k]).strip()]

    modifier = session.custom_style_modifier if session.style_id == "custom" else style.prompt_modifier
    if modifier:
        parts.append(modifier)

    prompt = ", ".join(parts)
    if not prompt:
        # Mode 2 permits an empty checklist; give the backend something workable
        # rather than an empty string it can't condition on.
        prompt = "a simple 3D prop, neutral design"
    return prompt


def _build_gen_request(session: Session, paths: dict) -> None:
    """Assemble the shape-stage request and checkpoint it.

    A single image becomes a MULTIVIEW request with only FRONT populated, which
    is what the current adapters receive. The request exists as a first-class,
    validated, resumable object — extra slots and control signals slot into it
    cleanly.

    Text-driven runs have no conditioning image, so they carry no request; the
    text-capable backends read the prompt directly.
    """
    if session.three_d_input_kind != "image":
        return

    image = (
        session.uploaded_image_path if session.mode == "image_upload"
        else session.generated_image_path
    )
    if not image:
        return

    request = genrequest.GenRequest.from_single_image(image)
    try:
        request.validate()
    except genrequest.ModeConflict as exc:
        raise PipelineError(str(exc)) from exc

    paths["request"].parent.mkdir(parents=True, exist_ok=True)
    paths["request"].write_text(json.dumps(request.to_dict(), indent=2), encoding="utf-8")


def _load_gen_request(paths: dict):
    """Read back the checkpointed request, if this run has one."""
    path = paths["request"]
    if not path.exists():
        return None
    try:
        return genrequest.GenRequest.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        # A malformed sidecar must not sink a run that can proceed on image_path.
        return None


def _determine_3d_input(session: Session) -> str:
    """Step 6 — decide whether step 7 generates from an image or from text.

    Mode 2 (direct text) with an image-conditioned backend needs a concept image
    but step 5 only runs for text_to_image mode. When this happens, generate the
    image here so step 7 has something to condition on.
    """
    if session.mode == "image_upload":
        if not session.uploaded_image_path:
            raise PipelineError("Image upload mode selected but no image was uploaded.")
        return "image"
    if session.procedural:
        return "text"     # procedural mode always reasons from the text prompt
    if session.mode == "text_to_image":
        if not session.generated_image_path:
            raise PipelineError("Text→Image mode reached step 6 without a generated image.")
        return "image"
    # Mode 2 (direct text): image-conditioned backends need an image, so route
    # through image generation implicitly; the mock/text-capable ones don't.
    backend_accepts_text = SETTINGS.model_backend == "mock"
    if not backend_accepts_text:
        # Generate a concept image from the prompt so the image-conditioned
        # backend has something to work with — step 5 only runs for mode 1.
        if not session.generated_image_path:
            sid = session.session_id
            paths = _step_paths(session)
            events.log(sid, "Direct Text mode with image-conditioned backend — generating concept image implicitly")
            session.generated_image_path = str(
                image_gen.generate_image(session.final_prompt, paths["concept_image"], session_id=sid)
            )
            sessions.save_session(session)
        return "image"
    return "text"


def _run_procedural(session: Session, style) -> None:
    """Step 7, procedural branch (§2.6.3)."""
    sid = session.session_id
    procedural_gen.eject_for_procedural(session_id=sid)

    code = procedural_gen.generate_procedural_code(session.final_prompt, style, session_id=sid)
    (sessions.session_dir(sid) / "procedural.py").write_text(code, encoding="utf-8")

    session.scene_object_uuid = uuid.uuid4().hex
    script = procedural_gen.build_runner_script(
        code, session.scene_object_uuid,
        str(blender_bridge.blend_file_path()),
        save=not blender_bridge.EMBEDDED,
    )
    result = blender_bridge.run_script(script, session_id=sid)

    if result.get("skipped"):
        raise PipelineError(
            "Procedural mode needs Blender to execute the generated bpy code, but BLENDER_PATH "
            "isn't set. Set it in .env, or switch procedural mode off."
        )
    if not result.get("ok"):
        # §9 — surface the failure; never try to auto-repair generated bpy code.
        raise PipelineError(
            f"Generated procedural code failed: {result.get('error')}. "
            "Nothing partial was left in the scene. Try rephrasing, or switch procedural mode off."
        )

    session.final_mesh_path = f"in_scene:{result['object_name']}"
    session.blend_file_path = str(blender_bridge.blend_file_path())
    events.log(sid, f"Procedural object built in-scene: {result['object_name']} ({result.get('tris', 0)} tris)")
