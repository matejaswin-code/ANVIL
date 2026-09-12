"""HTTP layer.

Thin by design: it validates input, hands work to anvil_core, and streams
progress. No pipeline logic lives here — §1's "single linear script" stays in
state_machine.py, and this file is just a way to reach it from a browser.
"""
from __future__ import annotations

import asyncio
import functools
import queue
import shutil
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from anvil_core import (
    batch,
    blender_bridge,
    edit_loop,
    events,
    export as export_mod,
    gpu_check,
    llm,
    model_gen,
    postprocess,
    schemas,
    sessions,
    state_machine,
    styles,
    texturing,
)
from anvil_core.config import ASSETS_DIR, PROJECT_ROOT, SESSIONS_DIR, SETTINGS

app = FastAPI(title="ANVIL", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

FRONTEND_DIR = PROJECT_ROOT / "frontend"


class PipelineBusy(RuntimeError):
    """Raised when something tries to start a second pipeline concurrently."""


class PipelineGate:
    """One pipeline at a time, process-wide.

    The single-VRAM-resident policy (§2.5.5) only holds if exactly one pipeline
    is running. Guarding just /api/generate isn't enough — a batch run, a session
    resume, and a shape edit all drive the same pipeline, so every one of them
    has to pass through the same gate or two of them will fight over the slot.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holder: str | None = None
        self._kind: str | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._holder is not None

    @property
    def status(self) -> dict:
        with self._lock:
            return {"busy": self._holder is not None, "holder": self._holder, "kind": self._kind}

    def acquire(self, holder: str, kind: str) -> None:
        with self._lock:
            if self._holder is not None:
                raise PipelineBusy(
                    f"A {self._kind} is already running ({self._holder}). "
                    "Only one generation runs at a time — wait for it to finish."
                )
            self._holder = holder
            self._kind = kind

    def release(self, holder: str) -> None:
        with self._lock:
            # Only the current holder may release, so a late finisher can't free
            # a gate that something else has since taken.
            if self._holder == holder:
                self._holder = None
                self._kind = None


GATE = PipelineGate()
batch.BATCH.attach_gate(GATE)


# ---------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    mode: str = "text_to_image"
    style_id: str = "realistic"
    procedural: bool = False
    checklist: dict = Field(default_factory=dict)
    flags: dict = Field(default_factory=dict)
    image_path: str | None = None
    # Text→Image only: stop after the concept image so it can be reviewed before
    # the 3D half runs. Batch runs leave this off — nothing is watching to approve.
    preview_first: bool = False


class ImageReviewRequest(BaseModel):
    # None keeps the existing prompt; a string replaces it for the redraw.
    prompt: str | None = None


class EditRequest(BaseModel):
    session_id: str
    instruction: str
    confirmed: bool = False


class ExportRequest(BaseModel):
    session_id: str
    format: str = "fbx"
    target: str = "generic"


class QueueAddRequest(GenerateRequest):
    label: str = ""


class ChecklistChatRequest(BaseModel):
    mode: str = "text_to_image"
    message: str
    checklist: dict = Field(default_factory=dict)
    history: list[dict] = Field(default_factory=list)


class SettingsPatch(BaseModel):
    llm_provider: str | None = None
    llm_endpoint: str | None = None
    llm_model: str | None = None
    llm_vision_capable: bool | None = None
    model_backend: str | None = None
    image_backend: str | None = None
    blender_path: str | None = None


# ---------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health():
    cap = gpu_check.check_gpu_capability()
    return {
        "ok": True,
        "version": "1.0.0",
        "gpu": cap,
        "blender": {
            "available": blender_bridge.blender_available(),
            "embedded": blender_bridge.EMBEDDED,
            "blend_file": str(blender_bridge.blend_file_path()),
        },
        "backends": {
            "model": SETTINGS.model_backend,
            "image": SETTINGS.image_backend,
            "llm": f"{SETTINGS.llm_provider} ({SETTINGS.llm_model})",
        },
        "resident": __import__("anvil_core").RESOURCE_MANAGER.resident,
    }


@app.get("/api/llm/health")
def llm_health():
    return llm.health_check()


@app.get("/api/styles")
def get_styles():
    return {"styles": styles.all_styles(), "procedural_unsuited": sorted(styles.PROCEDURAL_UNSUITED)}


@app.get("/api/schemas")
def get_schemas():
    return {"modes": schemas.MODE_SCHEMAS, "steps": schemas.PIPELINE_STEPS}


@app.get("/api/settings")
def get_settings():
    return {"settings": SETTINGS.as_dict(), "model_backends": model_gen.BACKEND_INFO}


@app.post("/api/settings")
def patch_settings(patch: SettingsPatch):
    """Live settings change. A backend swap ejects the previous resident model
    before the new one loads — the eject/load lifecycle from §2.5.2."""
    from anvil_core.resource_manager import RESOURCE_MANAGER

    changed = []
    for key, value in patch.model_dump(exclude_none=True).items():
        if getattr(SETTINGS, key, None) != value:
            setattr(SETTINGS, key, value)
            changed.append(key)

    if any(k in changed for k in ("model_backend", "llm_provider", "llm_model", "image_backend")):
        RESOURCE_MANAGER.eject()
        events.emit("settings_changed", changed=changed)

    return {"ok": True, "changed": changed, "settings": SETTINGS.as_dict()}


@app.post("/api/checklist/chat")
def checklist_chat(req: ChecklistChatRequest):
    """Conversational alternative to filling the form by hand.

    Returns the fields the message filled in, the field still outstanding, and a
    reply to show. The caller merges `fields` into whatever it already had, so
    typing into the form and talking here stay in sync — both write to the same
    checklist, and neither clears the other's work.
    """
    schema = schemas.MODE_SCHEMAS.get(req.mode)
    if not schema:
        raise HTTPException(400, f"Unknown mode {req.mode!r}")

    fields = list(schema.get("mandatory", [])) + list(schema.get("optional", []))
    if not fields:
        raise HTTPException(400, f"Mode {req.mode!r} has no checklist to fill in.")

    try:
        result = llm.fill_checklist(
            req.checklist,
            req.message,
            fields,
            history=req.history,
            mandatory_keys=[f["key"] for f in schema.get("mandatory", [])],
        )
    except llm.LLMError as exc:
        # The reasoning model being unreachable is a normal, recoverable state —
        # the form still works without it, so this is a 503 the UI can explain,
        # not a 500 that reads as a bug.
        raise HTTPException(503, str(exc)) from exc

    # Only keys the mode actually defines; a hallucinated field name would
    # otherwise ride along into the checklist and reach the prompt builder.
    known = {f["key"] for f in fields}
    result["fields"] = {k: v for k, v in result["fields"].items() if k in known}
    return result


# ---------------------------------------------------------------------------
# progress stream
# ---------------------------------------------------------------------------

@app.get("/api/status")
def pipeline_status():
    """Full pipeline snapshot for polling clients.

    The SSE stream carries the same information as it happens; this answers the
    same question for a client that arrived late, reconnected, or simply prefers
    to poll. Backend residency and VRAM are joined in here because `status` is
    kept free of imports that would cycle back through `events`.
    """
    from anvil_core.resource_manager import RESOURCE_MANAGER
    from anvil_core.status import PIPELINE_STATUS

    snap = PIPELINE_STATUS.snapshot()
    snap["gate"] = GATE.status
    snap["busy"] = snap["gate"]["busy"]
    snap["resident"] = RESOURCE_MANAGER.resident
    snap["backends"] = {
        "model": SETTINGS.model_backend,
        "image": SETTINGS.image_backend,
        "llm": f"{SETTINGS.llm_provider} ({SETTINGS.llm_model})",
    }

    vram = {"used_gb": None, "total_gb": gpu_check.check_gpu_capability().get("total_vram_gb")}
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            # reserved, not allocated: the caching allocator holds freed blocks, and
            # that reserved figure is what actually constrains the next model load.
            vram["used_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
    except Exception:
        pass
    snap["vram"] = vram

    snap["queue"] = batch.BATCH.snapshot()
    return snap


@app.get("/api/events")
async def event_stream():
    """Server-sent events carrying live pipeline progress.

    Each subscriber gets its own small executor rather than sharing asyncio's
    default pool. A blocking queue.get() parks a thread for as long as it waits,
    and the default pool is shared with everything else that runs blocking work —
    a handful of open tabs would otherwise be enough to starve it.
    """
    q = events.subscribe()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anvil-sse")

    async def generator():
        loop = asyncio.get_running_loop()
        try:
            yield events.sse_format({"type": "connected"})
            while True:
                try:
                    evt = await loop.run_in_executor(pool, functools.partial(q.get, True, 20.0))
                    yield events.sse_format(evt)
                except queue.Empty:
                    yield ": keepalive\n\n"
        except asyncio.CancelledError:
            raise
        finally:
            events.unsubscribe(q)
            pool.shutdown(wait=False)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def _run_in_worker(holder: str, fn, *args, **kwargs) -> None:
    """Run a pipeline on a worker thread, always releasing the gate afterwards."""
    def target():
        try:
            fn(*args, **kwargs)
        except Exception as exc:
            events.emit("error", message=str(exc))
        finally:
            GATE.release(holder)
    threading.Thread(target=target, daemon=True).start()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    missing = schemas.missing_mandatory(req.mode, req.checklist)
    if missing:
        raise HTTPException(400, f"Missing mandatory fields: {', '.join(missing)}")
    if req.mode == "image_upload" and not req.image_path:
        raise HTTPException(400, "Image upload mode needs an uploaded image.")

    try:
        styles.get_style(req.style_id)
    except KeyError:
        raise HTTPException(400, f"Unknown style preset: {req.style_id}") from None

    session = schemas.Session(
        mode=req.mode,
        style_id=req.style_id,
        procedural=req.procedural,
        checklist=req.checklist,
        flags={**schemas.DEFAULT_FLAGS, **req.flags},
        uploaded_image_path=req.image_path,
    )

    try:
        GATE.acquire(session.session_id, "generation")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None

    try:
        sessions.save_session(session)
    except Exception:
        GATE.release(session.session_id)
        raise

    # Text→Image can pause on the concept image so it can be judged before the
    # expensive half runs. Only meaningful when an image is actually produced —
    # procedural mode writes bpy code and never makes one.
    stop_after = "image_generate" if (
        req.preview_first and req.mode == "text_to_image" and not req.procedural
    ) else None

    _run_in_worker(
        session.session_id, state_machine.run_generation_pipeline, session, stop_after=stop_after,
    )
    return {"session_id": session.session_id, "status": "running", "will_pause": bool(stop_after)}


@app.post("/api/session/{session_id}/image/regenerate")
def regenerate_image(session_id: str, req: ImageReviewRequest):
    """Redraw the concept image, optionally from an edited prompt.

    Re-runs only step 5 (force_from=5) and parks again, so a user can iterate on
    the picture as many times as they like before committing to 3D.
    """
    session = sessions.load_session(session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    if session.mode != "text_to_image" or session.procedural:
        raise HTTPException(400, "Only Text→Image generations have a concept image to redraw.")

    if req.prompt is not None and req.prompt.strip():
        session.final_prompt = req.prompt.strip()
        # build_prompt would otherwise overwrite the edit on the next run.
        session.prompt_override = True
        sessions.save_session(session)

    try:
        GATE.acquire(session_id, "image regeneration")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None

    _run_in_worker(
        session_id, state_machine.run_generation_pipeline, session,
        force_from=5, stop_after="image_generate",
    )
    return {"session_id": session_id, "status": "running"}


@app.post("/api/session/{session_id}/continue")
def continue_to_3d(session_id: str):
    """Accept the concept image and run the rest of the pipeline (steps 6-9)."""
    session = sessions.load_session(session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    if session.status != "awaiting_review":
        raise HTTPException(400, f"Session isn't awaiting review (status: {session.status}).")

    try:
        GATE.acquire(session_id, "generation")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None

    _run_in_worker(session_id, state_machine.run_generation_pipeline, session)
    return {"session_id": session_id, "status": "running"}


@app.get("/api/session/{session_id}")
def get_session(session_id: str):
    session = sessions.load_session(session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    data = session.as_dict()
    if session.final_mesh_path and not session.final_mesh_path.startswith("in_scene:"):
        data["stats"] = postprocess.mesh_stats(Path(session.final_mesh_path))
    return data


@app.get("/api/sessions")
def list_sessions(resumable: bool = True):
    return {
        "sessions": sessions.list_resumable_sessions() if resumable else sessions.list_all_sessions()
    }


@app.post("/api/session/{session_id}/resume")
def resume_session(session_id: str):
    session = sessions.load_session(session_id)
    if session is None:
        raise HTTPException(404, "No such session")

    try:
        GATE.acquire(session_id, "generation")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None

    resumed_from = session.current_step
    try:
        state_machine._verify_procedural_scene_on_resume(session)
    except Exception:
        GATE.release(session_id)
        raise

    _run_in_worker(session_id, state_machine.run_generation_pipeline, session)
    return {"session_id": session_id, "status": "running", "resumed_from_step": resumed_from}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    return {"deleted": sessions.delete_session(session_id)}


# ---------------------------------------------------------------------------
# edit loop
# ---------------------------------------------------------------------------

@app.post("/api/edit")
def edit(req: EditRequest):
    session = sessions.load_session(req.session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    if session.status != "done":
        raise HTTPException(400, "That session hasn't finished generating yet.")

    # A shape or new_model edit re-runs the whole generation pipeline, so this
    # has to hold the same gate a fresh generation does.
    try:
        GATE.acquire(req.session_id, "edit")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None

    try:
        return edit_loop.handle_edit(session, req.instruction, confirmed=req.confirmed)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        GATE.release(req.session_id)


@app.post("/api/undo")
def undo(req: EditRequest):
    session = sessions.load_session(req.session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    try:
        GATE.acquire(req.session_id, "undo")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None
    try:
        return edit_loop.undo(session)
    finally:
        GATE.release(req.session_id)


@app.post("/api/rig")
def rig(req: EditRequest):
    session = sessions.load_session(req.session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    try:
        GATE.acquire(req.session_id, "rigging")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None
    try:
        return edit_loop.handle_edit(session, "[Rig this]", confirmed=True)
    except Exception as exc:
        raise HTTPException(400, str(exc)) from None
    finally:
        GATE.release(req.session_id)


def _finish_action(session_id: str, *, error: str | None = None) -> None:
    """Close out a post-generation action (remesh, texture) in the status snapshot.

    Deliberately does NOT emit session_done/session_failed. Those mean "the
    generation pipeline finished", and the UI titles them accordingly — a failed
    remesh was being reported to the user as "Generation failed", which is both
    wrong and alarming. The action's own caller already surfaces its result, so
    this only needs to update the pollable status.
    """
    from anvil_core.status import PIPELINE_STATUS

    PIPELINE_STATUS.finish(session_id, error=error)


@app.get("/api/remesh/availability")
def remesh_availability():
    """Whether the [Remesh] action can run, and the reason when it can't."""
    return postprocess.remesh_availability()


@app.post("/api/remesh")
def remesh(req: EditRequest):
    """Quad-remesh an already-generated mesh.

    Offered as an action rather than a pipeline step: it adds a minute or more per
    asset, and quad topology only earns that on the assets you keep and edit.
    """
    session = sessions.load_session(req.session_id)
    if session is None:
        raise HTTPException(404, "No such session")

    mesh = session.final_mesh_path
    if not mesh or mesh.startswith("in_scene:"):
        raise HTTPException(
            400,
            "This session has no mesh file to remesh. Procedural results live in the "
            "Blender scene rather than on disk.",
        )
    mesh_path = Path(mesh)
    if not mesh_path.exists():
        raise HTTPException(400, f"Mesh file is missing: {mesh}")

    state = postprocess.remesh_availability()
    if not state["available"]:
        raise HTTPException(400, state["reason"])

    try:
        GATE.acquire(req.session_id, "remeshing")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None

    out_path = mesh_path.parent / "remeshed_mesh.glb"
    try:
        events.step_start(req.session_id, "remesh", 1, 1, "Quad remesh")
        stats = postprocess.remesh_asset(
            mesh_path, out_path, styles.get_style(session.style_id), session_id=req.session_id,
        )
        events.step_done(req.session_id, "remesh", 1, 1)
        session.final_mesh_path = stats["path"]
        sessions.save_session(session)
        _finish_action(req.session_id)
        return {"ok": True, **stats}
    except Exception as exc:
        _finish_action(req.session_id, error=str(exc))
        raise HTTPException(400, str(exc)) from None
    finally:
        GATE.release(req.session_id)


@app.get("/api/texture/availability")
def texture_availability():
    """Whether the [Add texture] action can run, and the reason when it can't.

    Queried by the UI so the control can be disabled with a specific explanation
    instead of offering an action that will fail once pressed.
    """
    return texturing.availability()


@app.post("/api/texture")
def texture(req: EditRequest):
    """Paint PBR textures onto an already-generated mesh.

    A separate action rather than a pipeline step: painting costs minutes on top
    of generation, and is worth spending on the assets worth keeping.
    """
    session = sessions.load_session(req.session_id)
    if session is None:
        raise HTTPException(404, "No such session")

    mesh = session.final_mesh_path
    if not mesh or mesh.startswith("in_scene:"):
        raise HTTPException(
            400,
            "This session has no mesh file to texture. Procedural results live in the "
            "Blender scene rather than on disk.",
        )
    mesh_path = Path(mesh)
    if not mesh_path.exists():
        raise HTTPException(400, f"Mesh file is missing: {mesh}")

    image = session.generated_image_path or session.uploaded_image_path
    if not image:
        raise HTTPException(
            400,
            "Texturing is image-conditioned and this session has no concept image. "
            "Generate in Text-to-image or Image-upload mode to texture the result.",
        )

    state = texturing.availability()
    if not state["available"]:
        raise HTTPException(400, state["reason"])

    try:
        GATE.acquire(req.session_id, "texturing")
    except PipelineBusy as exc:
        raise HTTPException(409, str(exc)) from None

    out_path = mesh_path.parent / "textured_mesh.glb"
    try:
        events.step_start(req.session_id, "texture", 1, 1, "Paint PBR textures")
        stats = texturing.texture_current_asset(
            mesh_path, Path(image), out_path, session_id=req.session_id,
        )
        events.step_done(req.session_id, "texture", 1, 1)
        session.final_mesh_path = stats["path"]
        sessions.save_session(session)
        _finish_action(req.session_id)
        return {"ok": True, **stats}
    except Exception as exc:
        _finish_action(req.session_id, error=str(exc))
        raise HTTPException(400, str(exc)) from None
    finally:
        GATE.release(req.session_id)


@app.post("/api/export")
def export(req: ExportRequest):
    session = sessions.load_session(req.session_id)
    if session is None:
        raise HTTPException(404, "No such session")
    object_name = edit_loop._object_name(session)
    try:
        result = export_mod.export_asset(
            req.session_id, object_name, req.format, req.target,
            mesh_path=session.final_mesh_path,
        )
    except export_mod.ExportError as exc:
        raise HTTPException(400, str(exc))

    # Only record a path that actually contains a valid export (§9)
    session.exported_paths[f"{req.format}_{req.target}"] = result["path"]
    sessions.save_session(session)
    return result


@app.get("/api/session/{session_id}/download/{filename}")
def download(session_id: str, filename: str):
    path = (SESSIONS_DIR / session_id / "exports" / filename).resolve()
    if not str(path).startswith(str((SESSIONS_DIR / session_id).resolve())):
        raise HTTPException(400, "Invalid path")
    if not path.exists():
        raise HTTPException(404, "No such export")
    return FileResponse(path, filename=filename)


@app.get("/api/session/{session_id}/mesh")
def session_mesh(session_id: str):
    """Serve the finished .glb so the browser viewport can render it."""
    session = sessions.load_session(session_id)
    if session is None or not session.final_mesh_path:
        raise HTTPException(404, "No mesh for that session")
    if session.final_mesh_path.startswith("in_scene:"):
        raise HTTPException(409, "This asset lives in the Blender scene, not in a file.")
    path = Path(session.final_mesh_path)
    if not path.exists():
        raise HTTPException(404, "Mesh file is missing from disk")
    return FileResponse(path, media_type="model/gltf-binary")


@app.get("/api/session/{session_id}/concept")
def session_concept(session_id: str):
    session = sessions.load_session(session_id)
    if session is None or not session.generated_image_path:
        raise HTTPException(404, "No concept image for that session")
    return FileResponse(session.generated_image_path, media_type="image/png")


# ---------------------------------------------------------------------------
# uploads
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.png").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg", ".webp"):
        raise HTTPException(400, "Upload a PNG, JPG, or WebP image.")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ASSETS_DIR / f"{uuid.uuid4().hex[:10]}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    return {"path": str(dest), "name": file.filename, "url": f"/api/asset/{dest.name}"}


@app.get("/api/asset/{name}")
def get_asset(name: str):
    path = (ASSETS_DIR / name).resolve()
    if not str(path).startswith(str(ASSETS_DIR.resolve())) or not path.exists():
        raise HTTPException(404, "No such asset")
    return FileResponse(path)


# ---------------------------------------------------------------------------
# batch queue (§2.12)
# ---------------------------------------------------------------------------

@app.get("/api/batch")
def batch_state():
    return batch.BATCH.snapshot()


@app.post("/api/batch/add")
def batch_add(req: QueueAddRequest):
    missing = schemas.missing_mandatory(req.mode, req.checklist)
    if missing:
        raise HTTPException(400, f"Missing mandatory fields: {', '.join(missing)}")
    if req.mode == "image_upload" and not req.image_path:
        raise HTTPException(400, "Image upload mode needs an uploaded image.")
    try:
        item = batch.BATCH.add(
            mode=req.mode, style_id=req.style_id, checklist=req.checklist,
            flags={**schemas.DEFAULT_FLAGS, **req.flags},
            procedural=req.procedural, image_path=req.image_path, label=req.label,
        )
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return {"item": item, "queue": batch.BATCH.snapshot()}


@app.delete("/api/batch/item/{queue_id}")
def batch_remove(queue_id: str):
    try:
        batch.BATCH.remove(queue_id)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return batch.BATCH.snapshot()


@app.post("/api/batch/item/{queue_id}/retry")
def batch_retry(queue_id: str):
    batch.BATCH.retry(queue_id)
    return batch.BATCH.snapshot()


@app.post("/api/batch/start")
def batch_start():
    try:
        batch_id = batch.BATCH.start()
    except (RuntimeError, PipelineBusy) as exc:
        raise HTTPException(409, str(exc)) from None
    return {"batch_id": batch_id, "queue": batch.BATCH.snapshot()}


@app.post("/api/batch/stop")
def batch_stop():
    batch.BATCH.request_stop()
    return batch.BATCH.snapshot()


@app.delete("/api/batch")
def batch_clear():
    try:
        batch.BATCH.clear()
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from None
    return batch.BATCH.snapshot()


# ---------------------------------------------------------------------------
# static frontend (mounted last so /api/* wins)
# ---------------------------------------------------------------------------

class _RevalidatingStatic(StaticFiles):
    """Serve the frontend with revalidation forced on every request.

    The default headers let a browser reuse app.js/styles.css from memory cache for
    the rest of the session without asking. That is invisible until the frontend
    changes, at which point the UI silently stays on the old files and the only cure
    is a hard refresh nobody thinks to try. `no-cache` still allows a 304 — the file
    is re-sent only when it has actually changed — so this costs a conditional
    request, not bandwidth.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


if FRONTEND_DIR.exists():
    app.mount("/", _RevalidatingStatic(directory=str(FRONTEND_DIR), html=True), name="frontend")


def main() -> None:
    import uvicorn
    gpu_check.check_gpu_capability()
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")


if __name__ == "__main__":
    main()
