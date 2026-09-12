"""Export (§2.8).

Two formats (FBX, GLB) × two targets (generic, unreal). The `unreal` target
exists because Unreal's axis and scale conventions differ from Blender's, and
baking that into the export is the difference between an asset that drops in
cleanly and one that needs a manual 100x rescale every single time.

Per §9, a partial/failed export must NOT be recorded in session.exported_paths —
pointing at a file that doesn't contain a valid export is worse than no record.
"""
from __future__ import annotations

from pathlib import Path

from . import blender_bridge, events
from .config import SESSIONS_DIR


class ExportError(RuntimeError):
    pass


FORMATS = ("fbx", "glb")
TARGETS = ("generic", "unreal")

TARGET_PRESETS = {
    "generic": {
        "axis_forward": "-Z", "axis_up": "Y",
        "global_scale": 1.0,
        "apply_unit_scale": True,
        "description": "Blender-native conventions, -Z forward / Y up.",
    },
    "unreal": {
        # Unreal is X-forward, Z-up, and works in centimetres.
        "axis_forward": "X", "axis_up": "Z",
        "global_scale": 1.0,
        "apply_unit_scale": True,
        "use_space_transform": True,
        "description": "X forward / Z up, unit scale applied so 1 Blender unit = 1m in Unreal.",
    },
}


EXPORT_SCRIPT = '''
import bpy, json

_result = {{"ok": False}}
try:
    obj_name  = {object_name!r}
    fmt       = {fmt!r}
    target    = {target!r}
    out_path  = r"{out_path}"

    obj = bpy.context.scene.objects.get(obj_name)
    if obj is None:
        raise RuntimeError("object " + obj_name + " isn't in the scene")

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    # Include the armature too, if this asset has been rigged
    for mod in obj.modifiers:
        if mod.type == "ARMATURE" and mod.object is not None:
            mod.object.select_set(True)

    if fmt == "fbx":
        kwargs = dict(
            filepath=out_path,
            use_selection=True,
            apply_unit_scale={apply_unit_scale},
            global_scale={global_scale},
            axis_forward={axis_forward!r},
            axis_up={axis_up!r},
            add_leaf_bones=False,
            bake_anim=False,
        )
        if target == "unreal":
            kwargs["use_space_transform"] = True
            kwargs["bake_space_transform"] = True
            kwargs["primary_bone_axis"] = "X"
            kwargs["secondary_bone_axis"] = "-Y"
        bpy.ops.export_scene.fbx(**kwargs)
    elif fmt == "glb":
        bpy.ops.export_scene.gltf(
            filepath=out_path,
            export_format="GLB",
            use_selection=True,
            export_yup=(target != "unreal"),
        )
    else:
        raise RuntimeError("unsupported format: " + fmt)

    import os
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError("exporter returned without error but wrote no usable file")

    _result = {{"ok": True, "path": out_path, "bytes": os.path.getsize(out_path)}}

except Exception as exc:
    _result = {{"ok": False, "error": "{{}}: {{}}".format(type(exc).__name__, exc)}}

print("ANVIL_RESULT_JSON:" + json.dumps(_result))
'''


def export_asset(
    session_id: str, object_name: str, fmt: str, target: str, mesh_path: str | None = None
) -> dict:
    """Export the scene object. Returns {"path": str, "bytes": int}."""
    fmt = fmt.lower().strip()
    target = target.lower().strip()
    if fmt not in FORMATS:
        raise ExportError(f"Unsupported format {fmt!r}. Choose one of: {', '.join(FORMATS)}.")
    if target not in TARGETS:
        raise ExportError(f"Unsupported target {target!r}. Choose one of: {', '.join(TARGETS)}.")

    out_dir = SESSIONS_DIR / session_id / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"export_{target}.{fmt}"

    preset = TARGET_PRESETS[target]
    events.log(session_id, f"Exporting {fmt.upper()} for target '{target}' — {preset['description']}")

    if blender_bridge.blender_available():
        script = EXPORT_SCRIPT.format(
            object_name=object_name,
            fmt=fmt,
            target=target,
            out_path=str(out_path).replace("\\", "\\\\"),
            apply_unit_scale=preset["apply_unit_scale"],
            global_scale=preset["global_scale"],
            axis_forward=preset["axis_forward"],
            axis_up=preset["axis_up"],
        )
        result = blender_bridge.run_script(script, session_id=session_id)
        if not result.get("ok"):
            # Leave exported_paths untouched — §9
            raise ExportError(result.get("error") or "Export failed inside Blender")
        return {"path": result["path"], "bytes": result.get("bytes", 0), "format": fmt, "target": target}

    # ---- no Blender: fall back to a direct trimesh conversion of the mesh file
    if not mesh_path or not Path(mesh_path).exists():
        raise ExportError(
            "Blender isn't configured (BLENDER_PATH unset) and there's no mesh file to convert. "
            "Set BLENDER_PATH in .env to export from the live scene."
        )
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        if target == "unreal":
            # Blender Z-up → Unreal Z-up with X forward: rotate -90° about Z.
            import numpy as np
            rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [0, 0, 1])
            mesh.apply_transform(rot)
        mesh.export(out_path)
    except Exception as exc:
        raise ExportError(f"Direct mesh export failed: {exc}") from exc

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise ExportError("Export wrote no usable file")

    return {"path": str(out_path), "bytes": out_path.stat().st_size, "format": fmt, "target": target}
