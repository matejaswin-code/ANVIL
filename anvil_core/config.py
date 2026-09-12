"""Configuration — loads .env and exposes typed settings.

Everything is local-first: paths resolve inside the project directory, and no
setting here ever points at a hosted service unless the user explicitly opts in
by filling a cloud provider key in .env.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSIONS_DIR = PROJECT_ROOT / "sessions"
ASSETS_DIR = PROJECT_ROOT / "assets"
LOG_DIR = PROJECT_ROOT / "logs"
MODELS_DIR = PROJECT_ROOT / "models"

# Downloaded model weights stay inside the project rather than the global cache in
# ~/.cache/huggingface. Set before anything imports huggingface_hub, which reads this
# at import time. setdefault so a real environment variable still wins, matching the
# precedence rule _load_dotenv() uses below.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))
# rembg (background removal, used to isolate the subject before image-to-3D) keeps its
# u2net weights here rather than in ~/.u2net, for the same reason.
os.environ.setdefault("U2NET_HOME", str(MODELS_DIR / "u2net"))
# Hunyuan3D resolves weights through its own cache var, not HF_HOME — left unset it
# downloads a second private copy of the same multi-GB checkpoint to ~/.cache/hy3dgen.
os.environ.setdefault("HY3DGEN_MODELS", str(MODELS_DIR / "hy3dgen"))

# PyTorch's caching allocator reserves address space and never hands it back, so a
# long session of load/eject cycles pushes Windows' commit charge toward its limit
# even though little is resident — 45 GB committed against 6 GB of working set is
# typical after a dozen generations. expandable_segments lets it release segments
# instead. Must be set before torch initialises CUDA, hence here rather than at the
# call site.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency on python-dotenv being importable yet)."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables always win over .env
        os.environ.setdefault(key, value)


_load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _get_bool(key: str, default: bool = False) -> bool:
    return _get(key, "true" if default else "false").lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    """Runtime settings. Mutable at runtime via the Settings panel (§2.5.4)."""

    # --- server ---
    host: str = field(default_factory=lambda: _get("ANVIL_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _get_int("ANVIL_PORT", 8765))

    # --- reasoning LLM (§2.5.3) ---
    llm_provider: str = field(default_factory=lambda: _get("LLM_PROVIDER", "lm_studio"))
    llm_endpoint: str = field(default_factory=lambda: _get("LLM_ENDPOINT", "http://127.0.0.1:1234/v1"))
    llm_model: str = field(default_factory=lambda: _get("LLM_MODEL", "local-model"))
    llm_vision_capable: bool = field(default_factory=lambda: _get_bool("LLM_VISION_CAPABLE", True))
    openrouter_api_key: str = field(default_factory=lambda: _get("OPENROUTER_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY"))

    # --- 3D generation backend (§2.5.1) ---
    model_backend: str = field(default_factory=lambda: _get("MODEL_BACKEND", "mock"))
    # Shape-quality knobs. octree_resolution is the one that decides how much surface
    # detail exists at all — decimation can throw detail away but never invent it, so
    # a low value here caps quality no matter what the style's triangle target is.
    # Cost is roughly cubic in the grid, so 512 is ~2.4x the memory of 384.
    mesh_octree_resolution: int = field(default_factory=lambda: _get_int("MESH_OCTREE_RESOLUTION", 384))
    mesh_inference_steps: int = field(default_factory=lambda: _get_int("MESH_INFERENCE_STEPS", 50))
    mesh_guidance: float = field(default_factory=lambda: _get_float("MESH_GUIDANCE", 5.0))
    trellis2_path: str = field(default_factory=lambda: _get("TRELLIS2_PATH"))
    hunyuan3d_path: str = field(default_factory=lambda: _get("HUNYUAN3D_PATH"))
    hunyuan3d_mv_model: str = field(default_factory=lambda: _get("HUNYUAN3D_MV_MODEL", "tencent/Hunyuan3D-2mv"))
    hunyuan3d_mv_subfolder: str = field(default_factory=lambda: _get("HUNYUAN3D_MV_SUBFOLDER", "hunyuan3d-dit-v2-mv-turbo"))

    # --- image generation ---
    image_backend: str = field(default_factory=lambda: _get("IMAGE_BACKEND", "mock"))
    comfyui_endpoint: str = field(default_factory=lambda: _get("COMFYUI_ENDPOINT", "http://127.0.0.1:8188"))
    image_model_id: str = field(default_factory=lambda: _get("IMAGE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0"))
    # Sampler settings are model-specific: distilled models (sdxl-turbo) want very few
    # steps and guidance ~0-2, while full models (SDXL base) need ~30 steps and ~7.
    # Wrong values don't error, they just produce noise or mush — hence separate knobs.
    image_size: int = field(default_factory=lambda: _get_int("IMAGE_SIZE", 1024))
    image_steps: int = field(default_factory=lambda: _get_int("IMAGE_STEPS", 30))
    image_guidance: float = field(default_factory=lambda: _get_float("IMAGE_GUIDANCE", 7.0))

    # --- rigging (§2.7) ---
    unirig_path: str = field(default_factory=lambda: _get("UNIRIG_PATH"))

    # --- postprocess ---
    autoremesher_path: str = field(default_factory=lambda: _get("AUTOREMESHER_PATH"))

    # --- blender (§2.9 import target) ---
    blender_path: str = field(default_factory=lambda: _get("BLENDER_PATH"))
    blend_file: str = field(default_factory=lambda: _get("BLEND_FILE", "anvil_scene.blend"))

    # --- capability guard (§2.11) ---
    min_vram_warning_gb: float = field(default_factory=lambda: _get_float("MIN_VRAM_WARNING_GB", 8.0))
    require_gpu: bool = field(default_factory=lambda: _get_bool("REQUIRE_GPU", False))

    # --- loop caps (§7, §9) ---
    max_checklist_turns: int = field(default_factory=lambda: _get_int("MAX_CHECKLIST_TURNS", 10))
    max_regenerate_count: int = field(default_factory=lambda: _get_int("MAX_REGENERATE_COUNT", 10))

    def as_dict(self) -> dict:
        d = asdict(self)
        # Never leak secrets to the frontend — report presence only.
        for secret in ("openrouter_api_key", "gemini_api_key"):
            d[secret] = bool(d[secret])
        return d

    def api_key_for_provider(self) -> str:
        if self.llm_provider == "openrouter":
            return self.openrouter_api_key
        if self.llm_provider == "gemini":
            return self.gemini_api_key
        return ""


SETTINGS = Settings()


def ensure_dirs() -> None:
    for d in (SESSIONS_DIR, ASSETS_DIR, LOG_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)


ensure_dirs()
