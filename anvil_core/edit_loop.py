"""Step 10 — the edit loop (§2.7.4, §6, §9).

Four categories, four very different costs:

  property  — bpy.ops dispatch, instant, undoable
  shape     — re-runs the generation pipeline from step 7
  new_model — a brand new session entirely
  rig       — ejects the 3D backend, runs UniRig, restores

This module never calls run_anvil_session() — shape edits re-run
run_generation_pipeline directly and new_model creates a fresh session, which is
what stops edits recursing back through the top-level entry point (§7).
"""
from __future__ import annotations

from pathlib import Path

from . import edit_ops, events, llm, postprocess, rigging, sessions, state_machine
from .schemas import Session


class EditError(RuntimeError):
    pass


def _object_name(session: Session) -> str:
    if session.final_mesh_path and session.final_mesh_path.startswith("in_scene:"):
        return session.final_mesh_path.split(":", 1)[1]
    return f"ANVIL_{session.session_id[:6]}"


def handle_edit(session: Session, instruction: str, *, confirmed: bool = False) -> dict:
    """Classify and apply one edit instruction.

    Returns a dict the UI renders directly:
      {"category", "message", "needs_confirmation"?, "stats"?, "session_id"?}
    """
    sid = session.session_id
    instruction = instruction.strip()
    if not instruction:
        raise EditError("Empty edit instruction")

    classification = llm.classify_edit(instruction, session_id=sid)
    category = classification["category"]
    events.log(sid, f"Edit classified as '{category}' (confidence {classification['confidence']:.2f}, "
                    f"via {classification['source']})")

    # Gate low-confidence expensive categories behind an explicit confirmation
    # rather than silently burning a regeneration on a misread (§9).
    if classification["needs_confirmation"] and not confirmed:
        return {
            "category": category,
            "needs_confirmation": True,
            "message": (
                f"That looks like a '{category}' edit, which means "
                + ("regenerating the mesh" if category == "shape" else "replacing the object entirely")
                + ". Confidence is low — confirm to go ahead."
            ),
        }

    if category == "property":
        return _handle_property(session, instruction)
    if category == "rig":
        return _handle_rig(session, instruction)
    if category == "shape":
        return _handle_shape(session, instruction)
    if category == "new_model":
        return _handle_new_model(session, instruction)
    raise EditError(f"Unhandled edit category: {category}")


# ---------------------------------------------------------------------------

def _handle_property(session: Session, instruction: str) -> dict:
    sid = session.session_id
    extracted = llm.extract_property_params(instruction, session_id=sid)
    result = edit_ops.apply_property_edit(
        _object_name(session), extracted["op"], extracted["params"], session_id=sid
    )
    session.edit_history.append({
        "instruction": instruction,
        "category": "property",
        "undoable": bool(result.get("undo")),
        "undo_params": result.get("undo"),
    })
    sessions.save_session(session)
    return {
        "category": "property",
        "message": f"Applied {extracted['op']} — {instruction}",
        "op": extracted["op"],
        "simulated": result.get("simulated", False),
        "stats": {"tris": result.get("tris", 0), "verts": result.get("verts", 0)},
    }


def _handle_rig(session: Session, instruction: str) -> dict:
    sid = session.session_id
    if not session.final_mesh_path or session.final_mesh_path.startswith("in_scene:"):
        raise EditError(
            "Rigging needs a mesh file, and this asset lives directly in the scene "
            "(procedural mode). Export it first, or regenerate without procedural mode."
        )
    out_path = sessions.session_dir(sid) / "rigged.glb"
    stats = rigging.rig_current_asset(Path(session.final_mesh_path), out_path, session_id=sid)
    session.final_mesh_path = str(out_path)
    session.edit_history.append({
        "instruction": instruction, "category": "rig", "undoable": False, "undo_params": None,
    })
    sessions.save_session(session)
    return {
        "category": "rig",
        "message": f"Rigged — UniRig predicted {stats['bone_count']} bones with skinning weights.",
        "stats": {"bones": stats["bone_count"]},
    }


def _handle_shape(session: Session, instruction: str) -> dict:
    """The geometry itself must change → re-run the pipeline from step 7.

    Capped at MAX_REGENERATE_COUNT so a user who keeps rejecting output gets a
    useful suggestion instead of an unbounded loop (§9).
    """
    from .config import SETTINGS

    if session.regenerate_count >= SETTINGS.max_regenerate_count:
        return {
            "category": "shape",
            "message": (
                f"That's {session.regenerate_count} regenerations in a row. Still not right? "
                "Try adjusting the checklist fields rather than regenerating again — "
                "the prompt is where the leverage is."
            ),
            "capped": True,
        }

    # Fold the instruction into the checklist so the new prompt actually reflects it
    existing = session.checklist.get("style_desc", "")
    session.checklist["style_desc"] = f"{existing}, {instruction}".strip(", ")
    session.regenerate_count += 1

    for step in ("build_prompt", "model_generate", "postprocess", "import_to_blender"):
        if step in session.completed_steps:
            session.completed_steps.remove(step)
    session.current_step = 3
    sessions.save_session(session)

    session.edit_history.append({
        "instruction": instruction, "category": "shape", "undoable": False, "undo_params": None,
    })
    state_machine.run_generation_pipeline(session, force_from=4)

    stats = postprocess.mesh_stats(Path(session.final_mesh_path)) if session.final_mesh_path else {}
    return {
        "category": "shape",
        "message": "Mesh regenerated to reflect the change.",
        "stats": stats,
    }


def _handle_new_model(session: Session, instruction: str) -> dict:
    """A completely different asset → a brand new session.

    Creates + runs the new session and returns; it does NOT loop back through
    run_anvil_session (§7).
    """
    new_session = Session(
        mode=session.mode,
        style_id=session.style_id,
        procedural=session.procedural,
        checklist={"object_type": instruction},
        flags=dict(session.flags),
    )
    sessions.save_session(new_session)
    events.log(session.session_id, f"Starting a fresh session {new_session.session_id} for: {instruction}")

    state_machine.run_generation_pipeline(new_session)

    stats = postprocess.mesh_stats(Path(new_session.final_mesh_path)) if new_session.final_mesh_path else {}
    return {
        "category": "new_model",
        "message": "Generated a fresh asset from scratch.",
        "session_id": new_session.session_id,
        "stats": stats,
    }


def undo(session: Session) -> dict:
    ok, message = edit_ops.undo_last_edit(
        _object_name(session), session.edit_history, session_id=session.session_id
    )
    sessions.save_session(session)
    return {"ok": ok, "message": message}
