"""Blender bridge.

Two execution modes, resolved at import time:

  * embedded  — we're running INSIDE Blender (the addon), so `bpy` is importable
                and scripts run in-process against the live scene.
  * external  — we're a standalone server, so scripts run via
                `blender --background <blend> --python <script>`.

Everything above this module writes bpy scripts and calls run_script(); neither
caller nor script needs to know which mode is active. When Blender isn't
configured at all, operations degrade to "recorded but not applied" and say so
clearly, rather than silently pretending to have touched a scene.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from . import events
from .config import PROJECT_ROOT, SETTINGS

try:
    import bpy  # noqa: F401
    # Importable does NOT mean embedded. `bpy` also ships as an ordinary PyPI wheel —
    # Hunyuan3D's texture module pulls it in — which makes it importable inside plain
    # CPython. Treating that as "we are running inside Blender" is actively harmful:
    # scripts then execute in-process, and Blender's API is not thread-safe, so a call
    # from a request worker takes the whole server down instead of raising.
    #
    # Blender's own interpreter reports the path to its binary; the wheel leaves it empty.
    EMBEDDED = bool(getattr(bpy.app, "binary_path", ""))
except ImportError:
    EMBEDDED = False

RESULT_PREFIX = "ANVIL_RESULT_JSON:"


class BlenderError(RuntimeError):
    pass


def blender_available() -> bool:
    if EMBEDDED:
        return True
    path = SETTINGS.blender_path
    return bool(path) and Path(path).exists()


def blend_file_path() -> Path:
    p = Path(SETTINGS.blend_file)
    return p if p.is_absolute() else PROJECT_ROOT / p


def run_script(script: str, *, session_id: str = "", open_blend: bool = True) -> dict:
    """Execute a bpy script and return whatever it printed after RESULT_PREFIX."""
    if EMBEDDED:
        return _run_embedded(script, session_id=session_id)
    return _run_external(script, session_id=session_id, open_blend=open_blend)


def _extract_result(stdout: str) -> dict:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                continue
    return {"ok": False, "error": "Blender script produced no parseable result"}


def _run_embedded(script: str, *, session_id: str = "") -> dict:
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    namespace: dict = {"__name__": "__anvil_script__"}
    try:
        with redirect_stdout(buf):
            exec(compile(script, "<anvil_bpy_script>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return _extract_result(buf.getvalue())


def _run_external(script: str, *, session_id: str = "", open_blend: bool = True) -> dict:
    exe = SETTINGS.blender_path
    if not exe or not Path(exe).exists():
        events.log(
            session_id,
            "BLENDER_PATH isn't set — the scene operation was recorded but not applied to a .blend",
            "warn",
        )
        return {"ok": False, "skipped": True, "error": "Blender not configured (BLENDER_PATH unset)"}

    blend = blend_file_path()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(script)
        script_path = f.name

    cmd = [exe, "--background"]
    if open_blend and blend.exists():
        cmd.append(str(blend))
    cmd += ["--python", script_path]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=900, check=False)
    except subprocess.TimeoutExpired as exc:
        raise BlenderError("Blender timed out after 15 minutes") from exc
    finally:
        Path(script_path).unlink(missing_ok=True)

    result = _extract_result(res.stdout)
    if not result.get("ok") and "error" not in result:
        result["error"] = (res.stderr or res.stdout)[-400:]
    return result


# ---------------------------------------------------------------------------
# step 9 — import the generated mesh into the scene
# ---------------------------------------------------------------------------

IMPORT_SCRIPT = '''
import bpy, json

_result = {{"ok": False}}
try:
    mesh_path = r"{mesh_path}"
    obj_uuid  = "{object_uuid}"
    obj_name  = "{object_name}"

    before = set(bpy.data.objects)
    if mesh_path.lower().endswith(".glb") or mesh_path.lower().endswith(".gltf"):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    elif mesh_path.lower().endswith(".obj"):
        bpy.ops.wm.obj_import(filepath=mesh_path)
    elif mesh_path.lower().endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=mesh_path)
    else:
        raise RuntimeError("unsupported mesh format: " + mesh_path)

    new_objects = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
    if not new_objects:
        raise RuntimeError("import produced no mesh objects")

    # Join multi-part imports into one tracked object
    if len(new_objects) > 1:
        bpy.ops.object.select_all(action="DESELECT")
        for o in new_objects:
            o.select_set(True)
        bpy.context.view_layer.objects.active = new_objects[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = new_objects[0]

    obj.name = obj_name
    obj["anvil_uuid"] = obj_uuid
    bpy.context.view_layer.objects.active = obj

    mesh = obj.data
    mesh.calc_loop_triangles()

    if {save_blend}:
        bpy.ops.wm.save_as_mainfile(filepath=r"{blend_file}")

    _result = {{
        "ok": True,
        "object_name": obj.name,
        "verts": len(mesh.vertices),
        "tris": len(mesh.loop_triangles),
    }}
except Exception as exc:
    _result = {{"ok": False, "error": "{{}}: {{}}".format(type(exc).__name__, exc)}}

print("ANVIL_RESULT_JSON:" + json.dumps(_result))
'''


def import_mesh(mesh_path: Path, object_uuid: str, object_name: str, *, session_id: str = "") -> dict:
    blend = blend_file_path()
    script = IMPORT_SCRIPT.format(
        mesh_path=str(mesh_path).replace("\\", "\\\\"),
        object_uuid=object_uuid,
        object_name=object_name,
        blend_file=str(blend).replace("\\", "\\\\"),
        save_blend="True" if not EMBEDDED else "False",
    )
    return run_script(script, session_id=session_id)


# ---------------------------------------------------------------------------
# procedural-mode scene verification (§7, §9 — diverged scene on resume)
# ---------------------------------------------------------------------------

VERIFY_SCRIPT = '''
import bpy, json
_result = {{"ok": False}}
try:
    obj = bpy.context.scene.objects.get("{object_name}")
    matches = obj is not None and obj.get("anvil_uuid") == "{object_uuid}"
    _result = {{
        "ok": True,
        "present": bool(matches),
        "blend_file": bpy.data.filepath,
    }}
except Exception as exc:
    _result = {{"ok": False, "error": str(exc)}}
print("ANVIL_RESULT_JSON:" + json.dumps(_result))
'''


def verify_scene_object(object_name: str, object_uuid: str, *, session_id: str = "") -> dict:
    script = VERIFY_SCRIPT.format(object_name=object_name, object_uuid=object_uuid)
    return run_script(script, session_id=session_id)
