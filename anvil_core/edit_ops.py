"""Property edits and Undo Last Edit (§2.9, §6).

Property edits are the cheap path: `extract_property_params()` turns the
instruction into structured params and these functions dispatch them straight to
bpy — no regeneration pipeline involved, so once extraction is reliable this path
is effectively instant.

Every apply returns `undo_params`, which is what makes undo exact rather than
approximate: scale/reposition undo inverts the *relative* change, while
recolor/rename undo restores the *prior absolute* value. Computing an "opposite"
for the latter would be wrong (§10).
"""
from __future__ import annotations

import json

from . import blender_bridge, events


class EditOpError(RuntimeError):
    pass


VALID_OPS = ("scale", "recolor", "reposition", "rename")


def validate_params(op: str, params: dict) -> dict:
    """Reject a malformed/inconsistent params dict loudly rather than no-op'ing
    silently — a silent no-op is indistinguishable from a bug (§10)."""
    if op not in VALID_OPS:
        raise EditOpError(f"Unknown property op: {op!r}")

    if op == "scale":
        factor = params.get("factor")
        if not isinstance(factor, (int, float)) or factor <= 0:
            raise EditOpError(f"scale needs a positive numeric 'factor', got {factor!r}")
        return {"factor": float(factor)}

    if op == "recolor":
        rgb = params.get("rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            raise EditOpError(f"recolor needs 'rgb' as a 3-element list, got {rgb!r}")
        if not all(isinstance(c, (int, float)) and 0.0 <= float(c) <= 1.0 for c in rgb):
            raise EditOpError(f"recolor 'rgb' components must each be 0.0-1.0, got {rgb!r}")
        return {"rgb": [float(c) for c in rgb]}

    if op == "reposition":
        delta = params.get("delta")
        if not isinstance(delta, (list, tuple)) or len(delta) != 3:
            raise EditOpError(f"reposition needs 'delta' as a 3-element list, got {delta!r}")
        if not all(isinstance(c, (int, float)) for c in delta):
            raise EditOpError(f"reposition 'delta' components must be numeric, got {delta!r}")
        return {"delta": [float(c) for c in delta]}

    name = params.get("name")
    if not isinstance(name, str) or not name.strip():
        raise EditOpError(f"rename needs a non-empty 'name', got {name!r}")
    return {"name": name.strip()}


APPLY_SCRIPT = '''
import bpy, json

_result = {{"ok": False}}
try:
    op     = {op!r}
    params = json.loads({params_json!r})
    obj    = bpy.context.scene.objects.get({object_name!r})
    if obj is None:
        raise RuntimeError("object {object_name!r} isn't in the scene")

    undo = {{}}

    if op == "scale":
        f = params["factor"]
        undo = {{"op": "scale", "params": {{"factor": 1.0 / f}}}}
        obj.scale = (obj.scale.x * f, obj.scale.y * f, obj.scale.z * f)

    elif op == "reposition":
        d = params["delta"]
        undo = {{"op": "reposition", "params": {{"delta": [-d[0], -d[1], -d[2]]}}}}
        obj.location = (obj.location.x + d[0], obj.location.y + d[1], obj.location.z + d[2])

    elif op == "recolor":
        rgb = params["rgb"]
        if obj.data.materials:
            mat = obj.data.materials[0]
        else:
            mat = bpy.data.materials.new(name="ANVIL_Material")
            mat.use_nodes = True
            obj.data.materials.append(mat)
        bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat.use_nodes else None
        if bsdf is None:
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
        prev = list(bsdf.inputs["Base Color"].default_value)[:3]
        undo = {{"op": "recolor", "params": {{"rgb": prev}}}}   # absolute restore, not an inverse
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)

    elif op == "rename":
        undo = {{"op": "rename", "params": {{"name": obj.name}}}}   # absolute restore
        obj.name = params["name"]

    else:
        raise RuntimeError("unknown op: " + str(op))

    mesh = obj.data
    mesh.calc_loop_triangles()

    if {save_blend}:
        bpy.ops.wm.save_as_mainfile(filepath=r"{blend_file}")

    _result = {{"ok": True, "object_name": obj.name, "undo": undo,
                "tris": len(mesh.loop_triangles), "verts": len(mesh.vertices)}}

except Exception as exc:
    _result = {{"ok": False, "error": "{{}}: {{}}".format(type(exc).__name__, exc)}}

print("ANVIL_RESULT_JSON:" + json.dumps(_result))
'''


def apply_property_edit(object_name: str, op: str, params: dict, *, session_id: str = "") -> dict:
    """Apply one property edit. Returns {"ok", "undo": {op, params}, ...}."""
    clean = validate_params(op, params)
    events.log(session_id, f"Property edit: {op} {clean}")

    script = APPLY_SCRIPT.format(
        op=op,
        params_json=json.dumps(clean),
        object_name=object_name,
        blend_file=str(blender_bridge.blend_file_path()).replace("\\", "\\\\"),
        save_blend="True" if not blender_bridge.EMBEDDED else "False",
    )
    result = blender_bridge.run_script(script, session_id=session_id)

    if result.get("skipped"):
        # No Blender configured — record the intent so undo history stays coherent.
        return {"ok": True, "simulated": True, "undo": _synthetic_undo(op, clean), "object_name": object_name}
    if not result.get("ok"):
        raise EditOpError(result.get("error") or "Property edit failed in Blender")
    return result


def _synthetic_undo(op: str, params: dict) -> dict:
    """Best-effort inverse when Blender isn't attached — relative ops invert
    cleanly; absolute ops can't be reconstructed, so they're marked un-undoable."""
    if op == "scale":
        return {"op": "scale", "params": {"factor": 1.0 / params["factor"]}}
    if op == "reposition":
        d = params["delta"]
        return {"op": "reposition", "params": {"delta": [-d[0], -d[1], -d[2]]}}
    return {}


def undo_last_edit(object_name: str, edit_history: list[dict], *, session_id: str = "") -> tuple[bool, str]:
    """Revert the most recent undoable edit.

    Returns (success, message). Returns False — with a message the UI shows —
    when history is empty or the last edit was shape/new_model/rig, rather than
    doing nothing in a way that's indistinguishable from a bug (§9).
    """
    if not edit_history:
        return False, "There's nothing to undo yet."

    last = edit_history[-1]
    if not last.get("undoable") or not last.get("undo_params"):
        return False, (
            f"The last edit ({last.get('category', 'unknown')}) isn't undoable — "
            "only property edits can be reverted."
        )

    undo = last["undo_params"]
    try:
        apply_property_edit(object_name, undo["op"], undo["params"], session_id=session_id)
    except EditOpError as exc:
        return False, f"Undo failed: {exc}"

    edit_history.pop()
    return True, "Edit undone — the property is back to its prior value."
