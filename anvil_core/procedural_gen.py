"""Procedural mode (§2.6) — the second, independent generation path.

Instead of a neural mesh generator, the LLM writes bpy construction code which
is executed inside Blender. That's why enabling it ejects the 3D backend from
VRAM entirely: nothing neural is running at all.

Per §9, generated code that errors or produces degenerate geometry surfaces a
clear error rather than leaving a broken partial object in the scene — and we
deliberately do NOT try to auto-repair generated bpy code.
"""
from __future__ import annotations

import re

from . import events, llm
from .resource_manager import RESOURCE_MANAGER


class ProceduralError(RuntimeError):
    pass


SYSTEM_PROMPT = """You write Blender Python (bpy) code that constructs a single 3D object.

Rules:
- Use only `bpy`, `bmesh`, `mathutils`, and `math`. No file I/O, no network, no imports beyond those.
- Build ONE object and leave it selected and active.
- Name the object exactly: ANVIL_OBJECT
- Work at roughly unit scale (largest dimension ~1.0 Blender unit), centred at the origin.
- Prefer primitives + transforms + boolean modifiers. Keep it clean hard-surface geometry.
- Do NOT delete existing scene objects. Do NOT change render or scene settings.
- Reply with ONLY the Python code. No markdown fences, no commentary.
"""

# Anything outside this set has no business in generated construction code.
_FORBIDDEN = (
    r"\bimport\s+(?!bpy|bmesh|mathutils|math\b)",
    r"\b__import__\b", r"\beval\b", r"\bexec\b", r"\bopen\s*\(",
    r"\bsubprocess\b", r"\bos\.", r"\bsys\.", r"\bshutil\b", r"\bsocket\b",
    r"\brequests\b", r"\burllib\b", r"\bbpy\.ops\.wm\.(save|open|quit)",
)


def _sanitise(code: str) -> str:
    code = re.sub(r"^```(?:python)?|```$", "", code.strip(), flags=re.MULTILINE).strip()
    for pattern in _FORBIDDEN:
        if re.search(pattern, code):
            raise ProceduralError(
                "Generated bpy code contained a disallowed operation and was rejected. "
                "Try rephrasing the prompt, or switch procedural mode off for this asset."
            )
    if "ANVIL_OBJECT" not in code:
        raise ProceduralError(
            "Generated bpy code never named its object ANVIL_OBJECT, so the result can't be "
            "tracked in the scene. Try regenerating."
        )
    return code


def generate_procedural_code(prompt: str, style, *, session_id: str = "") -> str:
    """Ask the LLM for bpy construction code. The LLM is the only VRAM consumer
    in this path — the 3D backend was already ejected when procedural mode was
    enabled (§2.6.2)."""
    events.log(session_id, "Asking the LLM for bpy construction code")
    user = (
        f"Build this object: {prompt}\n\n"
        f"Style target: {style.label}"
        + (f" — {style.prompt_modifier}" if style.prompt_modifier else "")
        + f"\nKeep it under roughly {style.target_tris} triangles."
    )
    code = llm.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        temperature=0.2, max_tokens=2000, session_id=session_id,
    )
    return _sanitise(code)


def build_runner_script(code: str, object_uuid: str, blend_file: str | None, save: bool = True) -> str:
    """Wrap generated code in a runner that validates the result before saving.

    The validation is the point: a boolean that failed and left a zero-volume
    mesh must be caught here, not discovered later when the user wonders why the
    scene looks empty.
    """
    return f'''
import bpy, sys, json

_result = {{"ok": False}}

try:
    # ---- generated construction code -------------------------------------
{_indent(code, 4)}
    # ----------------------------------------------------------------------

    obj = bpy.data.objects.get("ANVIL_OBJECT")
    if obj is None:
        raise RuntimeError("generated code did not create an object named ANVIL_OBJECT")

    bpy.context.view_layer.objects.active = obj
    mesh = obj.data
    if len(mesh.vertices) == 0 or len(mesh.polygons) == 0:
        raise RuntimeError("generated code produced degenerate geometry (no vertices or faces)")

    dims = obj.dimensions
    if dims.x <= 1e-6 and dims.y <= 1e-6 and dims.z <= 1e-6:
        raise RuntimeError("generated code produced a zero-volume object")

    obj["anvil_uuid"] = "{object_uuid}"

    _result = {{
        "ok": True,
        "object_name": obj.name,
        "verts": len(mesh.vertices),
        "tris": len(mesh.loop_triangles) if mesh.loop_triangles else len(mesh.polygons),
        "dimensions": [dims.x, dims.y, dims.z],
    }}

    if {str(bool(save and blend_file))}:
        bpy.ops.wm.save_as_mainfile(filepath=r"{blend_file or ''}")

except Exception as exc:
    _result = {{"ok": False, "error": "{{}}: {{}}".format(type(exc).__name__, exc)}}

print("ANVIL_RESULT_JSON:" + json.dumps(_result))
'''


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.splitlines())


def eject_for_procedural(session_id: str = "") -> None:
    """§2.6.2 — enabling procedural mode frees the 3D backend's VRAM immediately."""
    RESOURCE_MANAGER.eject(session_id=session_id)
    events.log(session_id, "3D backend ejected — procedural mode needs no neural mesh generator")
