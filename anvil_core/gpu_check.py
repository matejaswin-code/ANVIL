"""Startup GPU/VRAM capability check (§2.11).

Runs ONCE at startup — not per-session, not inside the pipeline. Its whole job is
to catch "this machine fundamentally can't run any of this" before the user gets
partway into a checklist interview and only then discovers generation is
impossible. It is NOT a VRAM tracker; that's resource_manager's job.
"""
from __future__ import annotations

import shutil
import subprocess

from .config import SETTINGS

_CACHED: dict | None = None


def _probe_torch() -> tuple[bool, float | None, str | None]:
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return False, None, None
    try:
        if not torch.cuda.is_available():
            return False, None, None
        props = torch.cuda.get_device_properties(0)
        return True, props.total_memory / (1024 ** 3), props.name
    except Exception:
        return False, None, None


def _probe_nvidia_smi() -> tuple[bool, float | None, str | None]:
    """Fallback for when torch isn't importable yet at startup."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False, None, None
    try:
        res = subprocess.run(
            [exe, "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if res.returncode != 0 or not res.stdout.strip():
            return False, None, None
        first = res.stdout.strip().splitlines()[0]
        mem_mb, _, name = first.partition(",")
        return True, float(mem_mb.strip()) / 1024.0, name.strip()
    except Exception:
        return False, None, None


def check_gpu_capability(force: bool = False) -> dict:
    """Returns {cuda_available, total_vram_gb, device_name, warning, blocked}.

    No CUDA GPU  -> blocked=True when REQUIRE_GPU is set, so generation is stopped
                    with a clear message rather than failing inside a model .load().
    Low VRAM     -> blocked=False plus a one-time non-blocking warning, since it
                    might still work depending on which backend/quant is selected.
    """
    global _CACHED
    if _CACHED is not None and not force:
        return _CACHED

    available, vram, name = _probe_torch()
    if not available:
        available, vram, name = _probe_nvidia_smi()

    warning = None
    blocked = False

    if not available:
        if SETTINGS.require_gpu:
            blocked = True
            warning = (
                "No CUDA GPU detected. Generation is blocked because REQUIRE_GPU=true. "
                "Set REQUIRE_GPU=false in .env to run with the mock backends for UI testing."
            )
        else:
            warning = (
                "No CUDA GPU detected — real 3D backends will not run. "
                "Mock backends are available so the pipeline and UI still work end to end."
            )
    elif vram is not None and vram < SETTINGS.min_vram_warning_gb:
        warning = (
            f"Detected {vram:.1f}GB VRAM, below the configured {SETTINGS.min_vram_warning_gb:.0f}GB minimum. "
            "This may still work depending on which backend and quantisation you select."
        )

    _CACHED = {
        "cuda_available": available,
        "total_vram_gb": round(vram, 2) if vram else None,
        "device_name": name,
        "warning": warning,
        "blocked": blocked,
    }
    return _CACHED
