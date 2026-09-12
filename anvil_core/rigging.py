"""Rigging (§2.7) — UniRig as an edit-phase option.

Deliberately NOT gated by style preset: §2.7.1 says `[Rig this]` is available on
every asset, and some fraction of rig attempts on non-articulated props are
expected to fail by design rather than being prevented by a "is this mesh
rig-shaped" pre-check we'd have to invent and would get wrong.

The eject/load/restore cycle (§2.7.3) matters: UniRig takes the VRAM slot, and
whatever 3D backend was resident before is restored afterwards, so the next
generation doesn't pay a cold-load penalty the user didn't ask for.
"""
from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from . import events
from .config import SETTINGS
from .resource_manager import RESOURCE_MANAGER


class RiggingError(RuntimeError):
    pass


class Rigger(ABC):
    """Plugin contract, mirroring Generator (§2.7.2) — one implementation today,
    but the seam is where it belongs if a second ever shows up."""

    name: str = "abstract"

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def rig(self, mesh_path: Path, out_path: Path) -> dict: ...

    @abstractmethod
    def unload(self) -> None: ...


class UniRigRigger(Rigger):
    name = "unirig"

    def __init__(self) -> None:
        self._loaded = False

    def load(self) -> None:
        repo = SETTINGS.unirig_path
        if not repo or not Path(repo).exists():
            raise RiggingError(
                f"UniRig isn't configured (UNIRIG_PATH={repo!r}). "
                "Clone UniRig, point UNIRIG_PATH at it in .env, and re-run. "
                "Rigging is the one feature with no mock equivalent — a fake skeleton "
                "would be worse than no skeleton."
            )
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def rig(self, mesh_path: Path, out_path: Path) -> dict:
        if not self._loaded:
            raise RiggingError("UniRig rig() called before load()")
        repo = Path(SETTINGS.unirig_path)
        script = repo / "run.py"
        if not script.exists():
            raise RiggingError(f"UniRig entry point not found at {script}")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            res = subprocess.run(
                ["python", str(script), "--input", str(mesh_path), "--output", str(out_path)],
                cwd=str(repo), capture_output=True, text=True, timeout=900, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RiggingError("UniRig timed out after 15 minutes") from exc

        if res.returncode != 0:
            raise RiggingError(f"UniRig failed (rc={res.returncode}): {res.stderr[-400:]}")
        if not out_path.exists():
            raise RiggingError("UniRig reported success but produced no output file")

        stats = _inspect_rig(out_path)
        # §9: a degenerate rig is an expected occasional outcome, surfaced clearly
        # rather than silently importing an empty armature.
        if stats["bone_count"] == 0:
            raise RiggingError(
                "UniRig predicted zero bones for this mesh. This is expected for props that "
                "aren't articulated — the mesh is unchanged and nothing was imported."
            )
        return stats


def _inspect_rig(path: Path) -> dict:
    try:
        import trimesh
        scene = trimesh.load(path)
        graph = getattr(scene, "graph", None)
        bone_count = max(0, len(graph.nodes) - 1) if graph is not None else 0
        return {"bone_count": bone_count, "path": str(path)}
    except Exception:
        return {"bone_count": 0, "path": str(path)}


def rig_current_asset(mesh_path: Path, out_path: Path, *, session_id: str = "") -> dict:
    """Eject → load UniRig → rig → restore previous backend (§2.7.3)."""
    previous = RESOURCE_MANAGER.resident
    events.log(session_id, f"Ejecting {previous.get('key') or 'nothing'} to make room for UniRig")

    def _loader() -> UniRigRigger:
        rigger = UniRigRigger()
        rigger.load()
        return rigger

    rigger: UniRigRigger = RESOURCE_MANAGER.ensure_loaded(
        kind="rigger", key="unirig",
        loader=_loader, unloader=lambda r: r.unload(),
        session_id=session_id,
    )

    try:
        stats = rigger.rig(mesh_path, out_path)
    finally:
        # Restore whatever held the slot before, so the next generation is warm.
        RESOURCE_MANAGER.eject(session_id=session_id)
        if previous.get("kind") == "model_gen" and previous.get("key"):
            events.log(session_id, f"Restoring previous 3D backend: {previous['key']}")
            try:
                RESOURCE_MANAGER.ensure_loaded(
                    kind="model_gen", key=previous["key"],
                    loader=lambda: _reload_generator(previous["key"]),
                    unloader=lambda g: g.unload(),
                    session_id=session_id,
                )
            except Exception as exc:
                events.log(session_id, f"Couldn't restore previous backend ({exc}) — it will load on next use", "warn")

    return stats


def _reload_generator(backend_id: str):
    from . import model_gen
    gen = model_gen.REGISTRY[backend_id]()
    gen.load()
    return gen
