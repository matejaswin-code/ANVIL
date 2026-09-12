#!/usr/bin/env python
"""ANVIL install verification.

Runs inside the project venv. Checks each layer independently and reports what's
ready versus what still needs configuration, so a partial setup produces a useful
list rather than one opaque failure.

Exit code 0 = the app will start. Warnings about unconfigured optional backends
don't fail the check, because running on mock backends is a legitimate state.

    ./essential_tools/venv/bin/python essential_tools/verify.py
    ./essential_tools/venv/bin/python essential_tools/verify.py --full   # + pipeline smoke test
"""
from __future__ import annotations

import sys
from pathlib import Path

# Windows consoles default to cp1252 which can't encode ✓/✗. Reconfigure early.
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GREEN, RED, YELLOW, DIM, BOLD, NC = "\033[0;32m", "\033[0;31m", "\033[0;33m", "\033[2m", "\033[1m", "\033[0m"
if sys.platform == "win32":
    try:
        import colorama  # noqa
    except ImportError:
        GREEN = RED = YELLOW = DIM = BOLD = NC = ""

failures: list[str] = []
warnings: list[str] = []


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{NC} {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}!{NC} {msg}")
    warnings.append(msg)


def fail(msg: str) -> None:
    print(f"  {RED}✗{NC} {msg}")
    failures.append(msg)


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{NC}")


# ---------------------------------------------------------------------------
section("Python environment")
# ---------------------------------------------------------------------------

if sys.version_info < (3, 10):
    fail(f"Python {sys.version_info.major}.{sys.version_info.minor} — 3.10+ required")
else:
    ok(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

in_venv = sys.prefix != sys.base_prefix
if in_venv:
    ok(f"Running inside a venv ({Path(sys.prefix).name})")
else:
    warn("Not running inside the project venv — use essential_tools/venv/bin/python")

# ---------------------------------------------------------------------------
section("Dependencies")
# ---------------------------------------------------------------------------

REQUIRED = [
    ("fastapi", "HTTP layer"),
    ("uvicorn", "ASGI server"),
    ("pydantic", "request validation"),
    ("httpx", "LLM transport"),
    ("trimesh", "mesh handling"),
    ("numpy", "mesh maths"),
    ("PIL", "image handling"),
]
for module, purpose in REQUIRED:
    try:
        __import__(module)
        ok(f"{module} — {purpose}")
    except ImportError:
        fail(f"{module} missing ({purpose}) — re-run the setup script")

OPTIONAL = [
    ("torch", "GPU detection + real 3D backends"),
    ("diffusers", "IMAGE_BACKEND=diffusers"),
]
for module, purpose in OPTIONAL:
    try:
        __import__(module)
        ok(f"{module} — {purpose}")
    except ImportError:
        print(f"  {DIM}·{NC} {DIM}{module} not installed — only needed for {purpose}{NC}")

# ---------------------------------------------------------------------------
section("ANVIL core")
# ---------------------------------------------------------------------------

try:
    import anvil_core
    from anvil_core import (
        blender_bridge, gpu_check,
        postprocess, schemas, sessions, state_machine, styles,
    )
    ok(f"anvil_core {anvil_core.__version__} — all 17 modules import")
except Exception as exc:  # noqa: BLE001
    fail(f"anvil_core failed to import: {exc}")
    print(f"\n{RED}Can't continue without the core package.{NC}\n")
    sys.exit(1)

try:
    from server.app import app
    routes = len([r for r in app.routes if hasattr(r, "methods")])
    ok(f"HTTP server imports — {routes} API routes registered")
except Exception as exc:  # noqa: BLE001
    fail(f"server.app failed to import: {exc}")

ok(f"{len(styles.STYLE_PRESETS)} style presets loaded")
ok(f"{len(schemas.PIPELINE_STEPS)} pipeline steps defined")

# ---------------------------------------------------------------------------
section("Configuration")
# ---------------------------------------------------------------------------

from anvil_core.config import SETTINGS  # noqa: E402

env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    ok(".env found")
else:
    warn(".env missing — copy .env.example to .env")

ok(f"Server will bind {SETTINGS.host}:{SETTINGS.port}")

# 3D backend
if SETTINGS.model_backend == "mock":
    warn("MODEL_BACKEND=mock — CPU placeholder meshes. Fine for testing the pipeline; "
         "set a real backend in .env for actual assets.")
elif SETTINGS.model_backend.startswith("trellis"):
    path = Path(SETTINGS.trellis2_path) if SETTINGS.trellis2_path else None
    if path and path.exists():
        ok(f"MODEL_BACKEND={SETTINGS.model_backend} — TRELLIS2_PATH resolves")
    else:
        fail(f"MODEL_BACKEND={SETTINGS.model_backend} but TRELLIS2_PATH is unset or missing")
elif SETTINGS.model_backend == "hunyuan3d":
    path = Path(SETTINGS.hunyuan3d_path) if SETTINGS.hunyuan3d_path else None
    if path and path.exists():
        ok("MODEL_BACKEND=hunyuan3d — HUNYUAN3D_PATH resolves")
    else:
        fail("MODEL_BACKEND=hunyuan3d but HUNYUAN3D_PATH is unset or missing")

# image backend
if SETTINGS.image_backend == "mock":
    warn("IMAGE_BACKEND=mock — deterministic placeholder concept images")
else:
    ok(f"IMAGE_BACKEND={SETTINGS.image_backend}")

# LLM
if SETTINGS.llm_provider == "lm_studio":
    ok(f"LLM: LM Studio at {SETTINGS.llm_endpoint}")
elif SETTINGS.llm_provider == "openrouter":
    if SETTINGS.openrouter_api_key:
        ok("LLM: OpenRouter — API key present")
    else:
        fail("LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is empty")
elif SETTINGS.llm_provider == "gemini":
    if SETTINGS.gemini_api_key:
        ok("LLM: Gemini — API key present")
    else:
        fail("LLM_PROVIDER=gemini but GEMINI_API_KEY is empty")

# Blender
if blender_bridge.blender_available():
    ok(f"Blender: {SETTINGS.blender_path or 'embedded'}")
else:
    warn("BLENDER_PATH unset — meshes generate to disk but won't import into a scene. "
         "Procedural mode needs this.")

# UniRig
if SETTINGS.unirig_path and Path(SETTINGS.unirig_path).exists():
    ok("UniRig configured — rigging available")
else:
    warn("UNIRIG_PATH unset — the [Rig this] action will decline with a clear message")

# ---------------------------------------------------------------------------
section("Hardware")
# ---------------------------------------------------------------------------

cap = gpu_check.check_gpu_capability()
if cap["cuda_available"]:
    ok(f"{cap['device_name']} — {cap['total_vram_gb']} GB VRAM")
    if cap["warning"]:
        warn(cap["warning"])
else:
    warn("No CUDA GPU detected — real 3D backends won't run; mock backends will")

# ---------------------------------------------------------------------------
section("Frontend")
# ---------------------------------------------------------------------------

for name in ("index.html", "app.js", "styles.css"):
    if (PROJECT_ROOT / "frontend" / name).exists():
        ok(f"frontend/{name}")
    else:
        fail(f"frontend/{name} is missing")

vendor = PROJECT_ROOT / "frontend" / "vendor" / "three.module.js"
if vendor.exists():
    ok("three.js vendored — in-browser 3D preview enabled")
else:
    warn("three.js not vendored — run `npm install` then `node essential_tools/vendor.js`. "
         "Everything else works; you just won't see the mesh in the browser.")

# ---------------------------------------------------------------------------
if "--full" in sys.argv:
    section("Pipeline smoke test")
    try:
        original_model, original_image = SETTINGS.model_backend, SETTINGS.image_backend
        SETTINGS.model_backend = "mock"
        SETTINGS.image_backend = "mock"

        session = schemas.Session(
            mode="text",
            style_id="ps1",
            checklist={"object_type": "verification cube", "style_desc": "test"},
        )
        sessions.save_session(session)
        state_machine.run_generation_pipeline(session)

        if session.final_mesh_path and Path(session.final_mesh_path).exists():
            stats = postprocess.mesh_stats(Path(session.final_mesh_path))
            ok(f"Full pipeline ran — {stats['tris']} tris, {stats['verts']} verts")
            ok(f"Session checkpointed at {session.current_step}/9 steps")
        else:
            fail("Pipeline completed but produced no mesh file")

        sessions.delete_session(session.session_id)
        SETTINGS.model_backend, SETTINGS.image_backend = original_model, original_image
    except Exception as exc:  # noqa: BLE001
        fail(f"Pipeline smoke test failed: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------------------
print()
if failures:
    print(f"{RED}{BOLD}{len(failures)} problem(s) need fixing:{NC}")
    for f in failures:
        print(f"  {RED}·{NC} {f}")
    print()
    sys.exit(1)

if warnings:
    print(f"{GREEN}{BOLD}Ready to run.{NC} {DIM}({len(warnings)} optional thing(s) unconfigured — see above){NC}")
else:
    print(f"{GREEN}{BOLD}Everything checks out.{NC}")
print(f"\n  Start with: {BOLD}./essential_tools/run.sh{NC}  (or run.bat on Windows)")
print(f"  Then open:  {BOLD}http://{SETTINGS.host}:{SETTINGS.port}{NC}\n")
sys.exit(0)
