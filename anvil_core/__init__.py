"""ANVIL — local-first AI pipeline: text or image prompt → 3D model, into Blender.

Public surface. Everything a caller (the HTTP server, the Blender addon, a test)
should need is re-exported here; the submodules are the implementation.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .config import PROJECT_ROOT, SESSIONS_DIR, SETTINGS, Settings
from .gpu_check import check_gpu_capability
from .resource_manager import RESOURCE_MANAGER
from .schemas import MODE_SCHEMAS, PIPELINE_STEPS, Session, missing_mandatory
from .styles import STYLE_PRESETS, all_styles, get_style

__all__ = [
    "__version__",
    "PROJECT_ROOT", "SESSIONS_DIR", "SETTINGS", "Settings",
    "check_gpu_capability", "RESOURCE_MANAGER",
    "MODE_SCHEMAS", "PIPELINE_STEPS", "Session", "missing_mandatory",
    "STYLE_PRESETS", "all_styles", "get_style",
]


def run(prompt: str, style_id: str = "realistic", mode: str = "text") -> "Session":
    """Convenience one-liner for scripts and the CLI."""
    from .state_machine import run_anvil_session
    return run_anvil_session(
        None, mode=mode, style_id=style_id, checklist={"object_type": prompt}
    )
