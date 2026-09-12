"""Style-driven mesh postprocess (pipeline step 8).

Each function is independently testable against known geometry (a cube, a sphere,
a torus) — which is exactly how Milestone 2 says to validate them, because bugs
in decimation/voxelisation logic are far easier to spot on simple shapes than on
real generated output.

Every function takes and returns a trimesh.Trimesh, so the chain a style declares
in styles.py composes without any special-casing.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from . import events
from .config import SETTINGS


class PostprocessError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# individual operations
# ---------------------------------------------------------------------------

def remesh(mesh, style, session_id: str = ""):
    """Quad-remesh via AutoRemesher if it's available.

    Per §9: a failure here skips the step and flags the mesh as unremeshed rather
    than aborting the whole pipeline — but it must actually be flagged, so later
    steps don't assume clean quad topology they didn't get.
    """
    exe = SETTINGS.autoremesher_path or shutil.which("AutoRemesher") or shutil.which("autoremesher")
    if not exe or not Path(exe).exists():
        events.log(session_id, "AutoRemesher not configured — skipping remesh, mesh flagged unremeshed", "warn")
        mesh.metadata["anvil_unremeshed"] = True
        return mesh

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.obj"
        dst = Path(tmp) / "out.obj"
        mesh.export(src)
        try:
            res = subprocess.run(
                [str(exe), "-i", str(src), "-o", str(dst)],
                capture_output=True, text=True, timeout=600, check=False,
            )
            if res.returncode != 0 or not dst.exists():
                events.log(session_id, f"AutoRemesher failed (rc={res.returncode}) — mesh flagged unremeshed", "warn")
                mesh.metadata["anvil_unremeshed"] = True
                return mesh
            import trimesh
            out = trimesh.load(dst, force="mesh")
            out.metadata["anvil_unremeshed"] = False
            return out
        except (subprocess.TimeoutExpired, OSError) as exc:
            events.log(session_id, f"AutoRemesher errored ({exc}) — mesh flagged unremeshed", "warn")
            mesh.metadata["anvil_unremeshed"] = True
            return mesh


def decimate(mesh, style, session_id: str = ""):
    """Reduce triangle count toward the style's target_tris."""
    target = style.target_tris
    current = len(mesh.faces)
    if current <= target:
        events.log(session_id, f"Decimate skipped — {current} tris already under target {target}")
        return mesh
    try:
        simplified = mesh.simplify_quadric_decimation(face_count=target)
        if simplified is None or len(simplified.faces) == 0:
            raise PostprocessError("decimation produced an empty mesh")
        events.log(session_id, f"Decimated {current} -> {len(simplified.faces)} tris (target {target})")
        return simplified
    except Exception as exc:
        events.log(session_id, f"Decimation unavailable ({exc}) — keeping original topology", "warn")
        return mesh


def downsample_texture(mesh, style, session_id: str = ""):
    """Clamp texture resolution to the style's texture_size."""
    size = style.texture_size
    visual = getattr(mesh, "visual", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "image", None)
    if image is None:
        events.log(session_id, f"No texture image present — texture target {size}px recorded only")
        mesh.metadata["anvil_texture_target"] = size
        return mesh
    try:
        from PIL import Image
        if max(image.size) > size:
            material.image = image.resize((size, size), Image.NEAREST if size <= 128 else Image.LANCZOS)
            events.log(session_id, f"Texture downsampled to {size}px")
    except Exception as exc:
        events.log(session_id, f"Texture downsample skipped ({exc})", "warn")
    return mesh


def voxelize(mesh, style, session_id: str = ""):
    """Convert to blocky cubic volumes — the Gmod/Minecraft look.

    Triangle count is controlled by voxel *resolution*, not by decimating the
    result afterwards. Each surface voxel contributes 12 triangles, and quadric
    decimation on box-soup destroys exactly the hard axis-aligned faces that
    make the style read as blocky. So we search downward for the finest grid
    that still fits the style's budget, rather than over-generating and then
    grinding the corners off.
    """
    try:
        target = style.target_tris
        start = 24 if style.id == "minecraft_blocky" else 40
        best = None

        for divisions in range(start, 3, -2):
            pitch = max(mesh.extents) / divisions
            try:
                candidate = mesh.voxelized(pitch=pitch).as_boxes()
            except Exception:
                continue
            if candidate is None or len(candidate.faces) == 0:
                continue
            best = (candidate, divisions, pitch)
            if len(candidate.faces) <= target:
                break

        if best is None:
            raise PostprocessError("voxelisation produced no usable mesh at any resolution")

        out, divisions, pitch = best
        events.log(
            session_id,
            f"Voxelised at {divisions} divisions (pitch {pitch:.4f}) — "
            f"{len(out.faces)} tris against a {target} budget",
        )
        return out
    except Exception as exc:
        events.log(session_id, f"Voxelisation failed ({exc}) — keeping original mesh", "warn")
        return mesh


def flatten_normals(mesh, style, session_id: str = ""):
    """Kill smooth shading — faceted/flat look for the retro and toon presets."""
    try:
        mesh.unmerge_vertices()   # per-face vertices == no normal averaging
        events.log(session_id, "Normals flattened (faceted shading)")
    except Exception as exc:
        events.log(session_id, f"Normal flattening skipped ({exc})", "warn")
    return mesh


DISPATCH = {
    "remesh": remesh,
    "decimate": decimate,
    "downsample_texture": downsample_texture,
    "voxelize": voxelize,
    "flatten_normals": flatten_normals,
}


# ---------------------------------------------------------------------------
# chain runner
# ---------------------------------------------------------------------------

def run_postprocess(raw_path: Path, out_path: Path, style, *, session_id: str = "") -> Path:
    """Run the style's declared chain. `normal` declares an empty chain, so it is
    a genuine no-op rather than a special case in the code."""
    import trimesh

    mesh = trimesh.load(raw_path, force="mesh")
    if mesh is None or len(mesh.faces) == 0:
        raise PostprocessError(f"Can't postprocess {raw_path} — mesh is empty or unreadable")

    if not style.postprocess:
        events.log(session_id, f"Style '{style.id}' declares no postprocess — passing raw output through")
    for op_name in style.postprocess:
        # Remeshing is offered as an action on the finished asset instead of running
        # here (§ the [Remesh] button). It adds a minute or more to every generation,
        # and it is only worth paying for on assets you keep — the same reasoning that
        # keeps texturing and rigging out of the pipeline. The mesh stays flagged
        # `unremeshed` until it is actually run, so nothing downstream assumes quads.
        if op_name == "remesh":
            continue
        fn = DISPATCH.get(op_name)
        if fn is None:
            events.log(session_id, f"Unknown postprocess op {op_name!r} — skipped", "warn")
            continue
        events.log(session_id, f"postprocess: {op_name}")
        mesh = fn(mesh, style, session_id=session_id)

    if "remesh" in style.postprocess:
        mesh.metadata["anvil_unremeshed"] = True

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    return out_path


def remesh_availability() -> dict:
    """Whether the [Remesh] action can run, and why not if it can't."""
    exe = SETTINGS.autoremesher_path or shutil.which("AutoRemesher") or shutil.which("autoremesher")
    if not exe or not Path(exe).exists():
        return {
            "available": False,
            "reason": (
                "AutoRemesher isn't configured. Download it from "
                "github.com/huxingyi/autoremesher and set AUTOREMESHER_PATH in .env."
            ),
        }
    return {"available": True, "reason": ""}


def remesh_asset(mesh_path: Path, out_path: Path, style, *, session_id: str = "") -> dict:
    """Quad-remesh a finished asset in place of the pipeline step.

    Ordering matters and is the reason this isn't simply 'the same step, later':
    AutoRemesher sizes its quads from the input's edge density, so running it on an
    already-decimated mesh collapses it (22k triangles came back as ~850 faces in
    testing, against ~44k from the raw mesh). Decimating afterwards to the style's
    target restores the intended budget, so the chain here is remesh → decimate,
    matching the order the pipeline used to run.
    """
    import trimesh  # noqa: PLC0415

    state = remesh_availability()
    if not state["available"]:
        raise PostprocessError(state["reason"])

    mesh = trimesh.load(mesh_path, force="mesh")
    if mesh is None or len(mesh.faces) == 0:
        raise PostprocessError(f"Can't remesh {mesh_path} — mesh is empty or unreadable")

    # Remeshing rebuilds topology from scratch, which discards UVs and therefore the
    # painted maps with them. Refusing is the only kind option: texturing costs
    # minutes, and silently throwing that away to save a click is a bad trade.
    if getattr(mesh.visual, "uv", None) is not None:
        raise PostprocessError(
            "This mesh is already textured, and remeshing would discard the texture — "
            "quad remeshing rebuilds the topology and its UVs from scratch. "
            "Remesh first, then texture."
        )

    before = len(mesh.faces)
    mesh = remesh(mesh, style, session_id=session_id)
    if mesh.metadata.get("anvil_unremeshed"):
        raise PostprocessError(
            "AutoRemesher couldn't remesh this model, so it is unchanged. It aborts on "
            "meshes that are heavily fragmented or full of holes — generated meshes with "
            "many disconnected shells are the usual case. Texturing and export still work "
            "on the triangle mesh."
        )

    if style is not None and "decimate" in getattr(style, "postprocess", ()):
        events.log(session_id, "postprocess: decimate")
        mesh = decimate(mesh, style, session_id=session_id)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)
    return {"path": str(out_path), "faces_before": before, "faces_after": len(mesh.faces)}


def mesh_stats(path: Path) -> dict:
    """Triangle/vertex counts for the viewport HUD."""
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        return {
            "tris": int(len(mesh.faces)),
            "verts": int(len(mesh.vertices)),
            "watertight": bool(mesh.is_watertight),
            "unremeshed": bool(mesh.metadata.get("anvil_unremeshed", False)),
        }
    except Exception:
        return {"tris": 0, "verts": 0, "watertight": False, "unremeshed": True}
