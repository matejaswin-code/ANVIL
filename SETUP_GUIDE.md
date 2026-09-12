# ANVIL — Setup Guide

Local-first AI pipeline: a text or image prompt goes in, a styled 3D model comes out, straight into your Blender scene.

Everything runs on your machine. Nothing installs globally — the entire environment lives inside this project directory and can be deleted by removing one folder.

---

## Table of contents

1. [System requirements & prerequisites](#1-system-requirements--prerequisites)
2. [Step-by-step installation](#2-step-by-step-installation)
3. [Configuration reference (`.env`)](#3-configuration-reference-env)
4. [API keys & model integration](#4-api-keys--model-integration) — including [texture painting](#44-texture-painting-optional-post-generation)
5. [Local verification](#5-local-verification)
6. [Using the Blender addon](#6-using-the-blender-addon)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. System requirements & prerequisites

### What you must install yourself

These two can't be installed by the setup script, because they're the runtimes the script itself depends on.

| Requirement | Minimum | Why | Where |
|---|---|---|---|
| **Python** | 3.10+ | Runs the entire pipeline and server | [python.org/downloads](https://www.python.org/downloads/) |
| **Node.js** | 18+ | Only vendors `three.js` for the in-browser 3D preview. **Optional** — skip it and everything else still works | [nodejs.org](https://nodejs.org/) |

> **Windows:** during Python installation, tick **"Add Python to PATH"**. Nearly every "python is not recognized" problem traces back to this checkbox.
>
> **Debian/Ubuntu:** you also need `sudo apt install python3-venv`, which Debian splits out of the base Python package.

### What the setup script installs for you, locally

FastAPI, Uvicorn, httpx, trimesh, NumPy, Pillow, fast-simplification, manifold3d — all into `essential_tools/venv/`. Plus `three.js` into `node_modules/`. Nothing lands outside this directory.

### Optional external tools

None of these are needed to start. ANVIL ships with **mock backends** that run the complete 10-step pipeline on CPU with zero model weights, so you can confirm everything is wired up before downloading a single gigabyte.

| Tool | Enables | Needed? |
|---|---|---|
| **Blender 4.0+** | Importing meshes into a scene, property edits, procedural mode | Strongly recommended — without it, meshes land on disk but never reach a scene |
| **CUDA GPU** (8GB+) | Real 3D generation backends | Only for real generation; mock runs anywhere |
| **LM Studio** | Local reasoning LLM | Or use a cloud key instead |
| **TRELLIS.2** / **Hunyuan3D** | Actual 3D mesh generation | Only when you move off mock |
| **ComfyUI** | Real concept-image generation | Only when you move off mock |
| **UniRig** | The `Rig this` button | Only if you need skeletons |
| **AutoRemesher** | Quad remeshing in postprocess | Optional — skipped gracefully, mesh flagged `unremeshed` |

### Hardware guidance

| VRAM | What runs |
|---|---|
| None / no GPU | Mock backends. Full pipeline, placeholder geometry. |
| 6–8 GB | `trellis2_gguf` (~6 GB). Use a cloud LLM to avoid competing for the card. |
| 12 GB | `trellis2_gguf` or `hunyuan3d` comfortably, plus a local LLM. |
| 16 GB+ | `trellis2_full` (~11 GB). |

ANVIL keeps **at most one model in VRAM at a time** and ejects the previous one before loading the next, so these figures are a ceiling per-step rather than a sum.

---

## 2. Step-by-step installation

### Step 1 — Get the project

```bash
cd anvil
```

### Step 2 — Run the setup script

**Linux / macOS:**
```bash
chmod +x essential_tools/setup.sh
./essential_tools/setup.sh
```

**Windows** (Command Prompt or PowerShell):
```bat
essential_tools\setup.bat
```

The script checks your prerequisites, creates `essential_tools/venv/`, installs the Python dependencies into it, installs and vendors `three.js` into the project, copies `.env.example` to `.env`, creates the runtime directories, and then verifies the whole install.

It's safe to re-run at any time — it reuses an existing venv and won't overwrite an existing `.env`.

### Step 3 — Start ANVIL

**Linux / macOS:**
```bash
./essential_tools/run.sh
```

**Windows:**
```bat
run.bat
```

Open **http://127.0.0.1:8765**.

At this point ANVIL is fully functional on mock backends. Fill in the four required fields in the setup panel, press Generate, and you should see a real mesh appear in the centre viewport. That confirms all ten steps are wired up correctly — before you've configured a single model.

### Activating the environment manually (optional)

`run.sh` / `run.bat` use the local venv automatically. To use it directly:

```bash
# Linux / macOS
source essential_tools/venv/bin/activate

# Windows
essential_tools\venv\Scripts\activate.bat
```

Or skip activation entirely and call the interpreter by path:

```bash
./essential_tools/venv/bin/python -m server.app
```

### Uninstalling

```bash
rm -rf essential_tools/venv node_modules frontend/vendor sessions assets logs .env
```

That's the complete footprint. Nothing was written anywhere else on your system.

---

## 3. Configuration reference (`.env`)

`setup.sh` / `setup.bat` create `.env` from `.env.example`. Every value has a working default, so ANVIL starts even if you change nothing.

Settings you change in the in-app **Settings** panel apply immediately but are **not** written back to `.env` — this file stays your persistent baseline.

Here is the complete file, copy-pasteable:

```ini
# ============================================================================
#  ANVIL — environment configuration
# ============================================================================

# ----------------------------------------------------------------------------
# SERVER
# ----------------------------------------------------------------------------

# Interface the local server binds to. Keep 127.0.0.1 unless you deliberately
# want other machines on your network to reach it — there is no auth layer.
ANVIL_HOST=127.0.0.1

# Port for the web UI and API. Change it if 8765 is already taken.
ANVIL_PORT=8765


# ----------------------------------------------------------------------------
# REASONING LLM
#
# Drives the checklist interview, style captioning, edit classification, and
# (in procedural mode) writes the bpy construction code.
# ----------------------------------------------------------------------------

# Which provider to use: lm_studio | openrouter | gemini
#   lm_studio  — fully local, no API key, occupies VRAM
#   openrouter — cloud, needs a key, frees local VRAM entirely
#   gemini     — cloud, needs a key, frees local VRAM entirely
LLM_PROVIDER=lm_studio

# OpenAI-compatible endpoint. Only used when LLM_PROVIDER=lm_studio.
LLM_ENDPOINT=http://127.0.0.1:1234/v1

# Model identifier.
#   lm_studio  -> whatever LM Studio reports (or just "local-model")
#   openrouter -> e.g. anthropic/claude-3.5-sonnet
#   gemini     -> e.g. gemini-2.0-flash
LLM_MODEL=local-model

# Set true only if the selected model can actually read images. When false,
# captioning a reference image fails immediately with a clear message rather
# than sending the image anyway and getting a confusing response back.
LLM_VISION_CAPABLE=true

# API keys — only needed for the matching provider. Leave blank otherwise.
OPENROUTER_API_KEY=
GEMINI_API_KEY=


# ----------------------------------------------------------------------------
# 3D GENERATION BACKEND
# ----------------------------------------------------------------------------

# Which backend: mock | trellis2_gguf | trellis2_full | hunyuan3d
#   mock           0 GB VRAM, CPU only, no weights. Produces a real .glb so the
#                  whole pipeline is testable before downloading anything.
#   trellis2_gguf  ~6 GB VRAM  — quantised, practical on 8-12 GB cards
#   trellis2_full  ~11 GB VRAM — full precision, needs 12 GB+
#   hunyuan3d      ~9 GB VRAM
MODEL_BACKEND=mock

# Absolute path to a cloned TRELLIS repository.
TRELLIS2_PATH=

# Absolute path to a cloned Hunyuan3D-2 repository.
HUNYUAN3D_PATH=


# ----------------------------------------------------------------------------
# IMAGE GENERATION  (Text→Image mode only, pipeline step 5)
# ----------------------------------------------------------------------------

# Which backend: mock | comfyui | diffusers
IMAGE_BACKEND=mock

# ComfyUI's HTTP address. Only used when IMAGE_BACKEND=comfyui.
COMFYUI_ENDPOINT=http://127.0.0.1:8188

# HuggingFace model id. Only used when IMAGE_BACKEND=diffusers. Weights download
# into models/hf/ inside this project on first use, not the global ~/.cache.
# Loaded fp16, so the model must publish an fp16 variant.
IMAGE_MODEL_ID=stabilityai/sdxl-turbo


# ----------------------------------------------------------------------------
# RIGGING
# ----------------------------------------------------------------------------

# Absolute path to a cloned UniRig repository. Powers the [Rig this] button.
# Leave blank to disable rigging — the button then declines with an explanation
# instead of failing mid-run.
UNIRIG_PATH=


# ----------------------------------------------------------------------------
# MESH POST-PROCESSING
# ----------------------------------------------------------------------------

# Absolute path to an AutoRemesher binary, for quad remeshing. Optional — if
# missing, the remesh step is skipped and the mesh is flagged `unremeshed`.
AUTOREMESHER_PATH=


# ----------------------------------------------------------------------------
# BLENDER
# ----------------------------------------------------------------------------

# Absolute path to your Blender executable.
#   Linux:    /usr/bin/blender
#   macOS:    /Applications/Blender.app/Contents/MacOS/Blender
#   Windows:  C:\Program Files\Blender Foundation\Blender 4.2\blender.exe
BLENDER_PATH=

# The .blend file ANVIL imports into. Relative paths resolve inside the project.
BLEND_FILE=anvil_scene.blend


# ----------------------------------------------------------------------------
# CAPABILITY GUARD
# ----------------------------------------------------------------------------

# Below this much VRAM, ANVIL shows a one-time non-blocking warning at startup.
MIN_VRAM_WARNING_GB=8

# Set true to hard-block generation when no CUDA GPU is found.
REQUIRE_GPU=false


# ----------------------------------------------------------------------------
# LOOP CAPS
# ----------------------------------------------------------------------------

# How many turns the checklist interview may take before offering to generate.
MAX_CHECKLIST_TURNS=10

# How many consecutive shape regenerations before suggesting a prompt change.
MAX_REGENERATE_COUNT=10
```

### Variable summary

| Variable | Purpose | Default |
|---|---|---|
| `ANVIL_HOST` | Bind interface — no auth layer, keep local | `127.0.0.1` |
| `ANVIL_PORT` | UI + API port | `8765` |
| `LLM_PROVIDER` | `lm_studio` / `openrouter` / `gemini` | `lm_studio` |
| `LLM_ENDPOINT` | OpenAI-compatible URL (LM Studio only) | `http://127.0.0.1:1234/v1` |
| `LLM_MODEL` | Model identifier | `local-model` |
| `LLM_VISION_CAPABLE` | Whether the model reads images | `true` |
| `OPENROUTER_API_KEY` | OpenRouter credential | *(blank)* |
| `GEMINI_API_KEY` | Gemini credential | *(blank)* |
| `MODEL_BACKEND` | 3D generator | `mock` |
| `TRELLIS2_PATH` | TRELLIS repo location | *(blank)* |
| `HUNYUAN3D_PATH` | Hunyuan3D repo location | *(blank)* |
| `IMAGE_BACKEND` | Concept-image generator | `mock` |
| `COMFYUI_ENDPOINT` | ComfyUI address | `http://127.0.0.1:8188` |
| `IMAGE_MODEL_ID` | Diffusers model id | `stabilityai/stable-diffusion-xl-base-1.0` |
| `IMAGE_SIZE` | Output resolution, square px | `1024` |
| `IMAGE_STEPS` | Denoising steps | `30` |
| `IMAGE_GUIDANCE` | Prompt adherence | `7.0` |
| `UNIRIG_PATH` | UniRig repo location | *(blank)* |
| `AUTOREMESHER_PATH` | AutoRemesher binary | *(blank)* |
| `BLENDER_PATH` | Blender executable | *(blank)* |
| `BLEND_FILE` | Target scene file | `anvil_scene.blend` |
| `MIN_VRAM_WARNING_GB` | Warning threshold | `8` |
| `REQUIRE_GPU` | Hard-block without CUDA | `false` |
| `MAX_CHECKLIST_TURNS` | Interview cap | `10` |
| `MAX_REGENERATE_COUNT` | Regeneration cap | `10` |

---

## 4. API keys & model integration

### 4.1 Reasoning LLM — pick one

The LLM handles the checklist interview, style captioning, edit classification, and procedural code generation. You need exactly one of these three.

#### Option A — LM Studio (local, free, no key)

1. Install [LM Studio](https://lmstudio.ai/).
2. Download a model. Anything in the 7B–31B range works well; larger models classify edits more reliably.
3. Open the **Developer** tab and **Start Server**. Note the port (default `1234`).
4. In `.env`:

```ini
LLM_PROVIDER=lm_studio
LLM_ENDPOINT=http://127.0.0.1:1234/v1
LLM_MODEL=local-model
LLM_VISION_CAPABLE=true    # only if your model actually reads images
```

> Set `LLM_VISION_CAPABLE=false` for a text-only model. ANVIL then refuses image captioning up front with a clear message instead of sending an image the model can't see.

**Trade-off:** occupies VRAM that the 3D backend also wants. On 8GB cards, a cloud LLM is often the better split.

#### Option B — OpenRouter (cloud, paid)

1. Sign up at [openrouter.ai](https://openrouter.ai/), then create a key at [openrouter.ai/keys](https://openrouter.ai/keys).
2. In `.env`:

```ini
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-3.5-sonnet
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

#### Option C — Gemini (cloud, generous free tier)

1. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. In `.env`:

```ini
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.0-flash
GEMINI_API_KEY=your-key-here
```

> Cloud providers use **no local VRAM at all**, freeing the entire card for 3D generation. Switching to one ejects any resident local model immediately.

**Verify it:** open **Settings → Test LLM** in the UI. You'll get either a confirmed reply or a specific reason it failed.

### 4.2 3D generation backend

#### TRELLIS.2 (recommended)

```bash
cd ~/models
git clone https://github.com/microsoft/TRELLIS
cd TRELLIS
# Install its dependencies into ANVIL's venv, not globally:
/path/to/anvil/essential_tools/venv/bin/pip install -r requirements.txt
```

Then in `.env`:

```ini
MODEL_BACKEND=trellis2_gguf        # ~6 GB VRAM
TRELLIS2_PATH=/home/you/models/TRELLIS
```

Use `trellis2_full` instead if you have 12GB+ of VRAM.

#### Hunyuan3D 2.1

Works on Windows, and needs **no compiler** — only the shape half of the repo is used.

```bash
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 backends/Hunyuan3D-2.1
```

Install the shape-path dependencies. Do **not** use the repo's own `requirements.txt`: it pins `diffusers==0.30.0`, `transformers==4.46.0`, and `numpy==1.24.4`, which downgrades the shared venv and breaks the diffusers image backend.

```bash
essential_tools\venv\Scripts\pip.exe install einops omegaconf scikit-image opencv-python pymeshlab pygltflib timm rembg onnxruntime
essential_tools\venv\Scripts\pip.exe install torchvision --index-url https://download.pytorch.org/whl/cu128
```

```ini
MODEL_BACKEND=hunyuan3d
HUNYUAN3D_PATH=C:\path\to\anvil\backends\Hunyuan3D-2.1
```

The ~6.9 GB shape checkpoint downloads on first use. It resolves through Hunyuan's own `HY3DGEN_MODELS` variable, not `HF_HOME`; ANVIL points that at `models/hy3dgen/` so it doesn't land in `~/.cache/hy3dgen`.

**Why no CUDA extensions are needed.** The repo's compiled pieces (`custom_rasterizer`, `DifferentiableRenderer`) live in `hy3dpaint/`, which does texture painting — ANVIL does its own texturing and never imports it. Three optional extensions in the shape path are all avoidable: `sageattention` is gated behind `USE_SAGEATTN=1` (off by default), `diso` is only used by `mc_algo='dmc'` (ANVIL passes `'mc'`, skimage's marching cubes), and `torch_cluster` is encoder-only, unused for image-to-3D.

**Input images must be object-centric.** The model reads the alpha channel to find the subject. A full-frame photo with no clear subject margin reconstructs as a flat slab rather than a solid — the mesh is watertight but paper-thin on one axis. ANVIL runs `rembg` automatically when the alpha carries no information, but the concept image still has to depict *one object with space around it*. Prompts like "centered product photo, isolated on a plain background" work; "a landscape with a tree" does not.

Expect roughly 3–4 minutes per mesh on a 12 GB card, at ~7.4 GB VRAM.

#### TRELLIS.2

Requires **24 GB of VRAM and Linux** ([README](https://github.com/microsoft/TRELLIS.2)), and its `setup.sh` builds six CUDA extensions. It will not run on a 12 GB Windows machine.

#### PyTorch

Real backends need PyTorch matching your CUDA version — install it into the project venv:

```bash
# Linux/macOS
./essential_tools/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124

# Windows
essential_tools\venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu124
```

Check your CUDA version with `nvidia-smi` and pick the matching index URL from [pytorch.org](https://pytorch.org/get-started/locally/).

> **RTX 50-series (Blackwell):** these cards report compute capability 12.0 (`sm_120`), which the `cu124` build above does not support — it installs fine and then fails at the first CUDA call. Use `cu128` or newer instead:
>
> ```bash
> essential_tools\venv\Scripts\pip.exe install torch --index-url https://download.pytorch.org/whl/cu128
> ```

### 4.3 Image generation (Text→Image mode)

**ComfyUI** — point ANVIL at a running instance:

```ini
IMAGE_BACKEND=comfyui
COMFYUI_ENDPOINT=http://127.0.0.1:8188
```

Export a workflow from ComfyUI via **File → Export (API)** and save it to `anvil_core/workflows/txt2img.json`. ANVIL injects the prompt into any `CLIPTextEncode` node whose title starts with "positive".

**Diffusers** — loads a model directly into VRAM, no second application to run. Needs PyTorch (§ above) plus:

```bash
./essential_tools/venv/bin/pip install diffusers transformers accelerate safetensors
```

```ini
IMAGE_BACKEND=diffusers
IMAGE_MODEL_ID=stabilityai/stable-diffusion-xl-base-1.0
IMAGE_SIZE=1024
IMAGE_STEPS=30
IMAGE_GUIDANCE=7.0
```

Weights download on first use into `models/hf/` inside the project — around 7 GB. To fetch them ahead of time rather than stalling your first generation:

```bash
./essential_tools/venv/bin/python -c "from anvil_core import config; from huggingface_hub import snapshot_download; snapshot_download('stabilityai/stable-diffusion-xl-base-1.0', allow_patterns=['*.json','*.txt','*/*fp16.safetensors'])"
```

Importing `anvil_core.config` first is what redirects the cache into the project — `huggingface_hub` reads `HF_HOME` once at import, so importing `diffusers` before it silently re-downloads everything to `~/.cache`.

Two things the `allow_patterns` filter does: it skips the fp32 weights, which are never loaded (ANVIL requests `variant="fp16"`), and the `*/` prefix restricts matches to the component subfolders. Without that prefix it also pulls the multi-GB single-file checkpoints at the repo root, which are for ComfyUI/A1111 and which diffusers never touches.

**Choosing a model.** `IMAGE_SIZE`, `IMAGE_STEPS`, and `IMAGE_GUIDANCE` are model-specific, and wrong values degrade quietly instead of erroring — a distilled model at 30 steps is mush, a full model at 8 steps is noise. Two known-good pairings:

| Model | Size | Steps | Guidance | Speed | Notes |
|---|---|---|---|---|---|
| `stabilityai/stable-diffusion-xl-base-1.0` | 1024 | 30 | 7.0 | ~10-20 s | Native 1024, best fidelity. OpenRAIL++ licence. |
| `stabilityai/sdxl-turbo` | 512 | 8 | 1.5 | ~2 s | Distilled, very fast. Repeats structure above 512. Non-commercial research licence. |

### 4.4 Texture painting (optional, post-generation)

The 3D backends produce untextured geometry. Hunyuan3D-2.1's paint half adds real PBR materials — base colour, metallic, roughness — projected from the concept image. It runs as an action on a finished asset (the **Texture** button next to *Rig* / *Export*), not as a pipeline step: it costs ~4-5 minutes per mesh, which is worth spending on the assets you keep rather than on every candidate.

If any prerequisite below is missing, the button stays disabled and names what's absent, so a half-finished install never fails mid-run.

**Prerequisites beyond the shape backend:**

| Requirement | Why |
|---|---|
| CUDA Toolkit matching PyTorch's build | `custom_rasterizer` is a CUDA extension; a major-version mismatch is a hard error |
| Visual Studio Build Tools, C++ workload | MSVC compiles both extensions |
| Blender (`BLENDER_PATH`) | Converts the painted OBJ to GLB |
| ~7 GB weights | `tencent/Hunyuan3D-2.1`, `hunyuan3d-paintpbr-v2-1` subfolder |

```bash
essential_tools\venv\Scripts\pip.exe install xatlas timm pytorch_lightning realesrgan pybind11 "bpy==5.2.0"
```

Fetch the upscaler checkpoint and the paint weights:

```bash
curl -L https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -o backends/Hunyuan3D-2.1/hy3dpaint/ckpt/RealESRGAN_x4plus.pth
essential_tools\venv\Scripts\python.exe -c "from anvil_core import config; from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', allow_patterns=['hunyuan3d-paintpbr-v2-1/*'])"
```

Then compile the two extensions. Both need the VC environment active **before** anything reads `%PATH%`, `DISTUTILS_USE_SDK=1`, and `CUDA_HOME` pointing at the toolkit matching PyTorch:

```bat
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=%CUDA_HOME%\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
cd backends\Hunyuan3D-2.1\hy3dpaint\DifferentiableRenderer
..\..\..\..\essential_tools\venv\Scripts\python.exe setup_windows.py build_ext --inplace
cd ..\custom_rasterizer
..\..\..\..\essential_tools\venv\Scripts\pip.exe install . --no-build-isolation
```

#### Source changes this needs on Windows

Hunyuan3D targets Linux, and several assumptions don't hold on Windows. Each patched file keeps a `.orig` backup beside it.

| File | Change | Why |
|---|---|---|
| `custom_rasterizer_kernel/grid_neighbor.cpp` | `(int64_t)` casts on 13 `torch::zeros({...})` dims | MSVC treats narrowing inside a braced initialiser as an error (C2398); GCC only warns |
| `…/*.cpp`, `*.cu` | `data_ptr<long>` → `data_ptr<int64_t>` (14 sites) | `long` is 32-bit on Windows, so that instantiation doesn't exist in libtorch — unresolved external at link |
| `custom_rasterizer_kernel/rasterizer.h` | `torch/extension.h` only outside `__CUDACC__` | Under nvcc it collides with CUDA's CCCL headers: `'std': ambiguous symbol` |
| `custom_rasterizer_kernel/rasterizer_gpu.cu` | z-buffer sentinel cast to `int64_t` | `(long)` truncated ~4.6e18 to 32 bits, so the face index recovered from the z-buffer pointed outside the vertex array — a CUDA *illegal memory access*, not a wrong image |
| `DifferentiableRenderer/setup_windows.py` | new file | The shipped `compile_mesh_painter.sh` assumes `g++` and `python3-config` |
| `hy3dpaint/utils/multiview_utils.py` | `trust_remote_code=True` | Loads `hy3dpaint/hunyuanpaintpbr` — local repo code, not a runtime download. Their pinned diffusers 0.30 didn't require the flag; 0.31+ does |

`basicsr` (pulled in by the upscaler) also fails to build on Python 3.13: its `setup.py` reads the version via `exec` into `locals()`, which PEP 667 made a snapshot. Install from a source copy with `get_version()` exec'ing into an explicit dict instead.

#### Two things that will crash the server if reintroduced

Both are handled in [texturing.py](anvil_core/texturing.py), and both are silent-death rather than exception:

- **`save_glb=False` is mandatory** when calling the paint pipeline. Its own OBJ→GLB step drives `bpy` in-process, and Blender's API needs the main thread — from a request worker it takes the process down. ANVIL converts out-of-process through `blender_bridge` instead.
- **Installing `bpy` changes how ANVIL talks to Blender.** `blender_bridge` decided it was running *inside* Blender based on `import bpy` succeeding, which the PyPI wheel makes true in ordinary CPython. Every Blender operation then ran in-process. It now checks `bpy.app.binary_path`, which is empty for the wheel and set inside Blender.

> Hunyuan3D is licensed for **non-commercial** use.

### 4.5 Rigging (UniRig)

```bash
git clone https://github.com/VAST-AI-Research/UniRig
```

```ini
UNIRIG_PATH=/home/you/models/UniRig
```

There is deliberately **no mock rigger** — a fake skeleton would be worse than no skeleton. With `UNIRIG_PATH` blank, the `Rig this` button declines with an explanation rather than failing mid-run.

### 4.6 Blender

Set the absolute path to your Blender executable:

```ini
BLENDER_PATH=/usr/bin/blender
BLEND_FILE=anvil_scene.blend
```

| Platform | Typical path |
|---|---|
| Linux | `/usr/bin/blender` or `/opt/blender/blender` |
| macOS | `/Applications/Blender.app/Contents/MacOS/Blender` |
| Windows | `C:\Program Files\Blender Foundation\Blender 4.2\blender.exe` |

Without it, meshes are still generated to disk and the UI tells you they weren't imported. **Procedural mode requires it**, since it executes generated `bpy` code.

---

## 5. Local verification

### Quick check

```bash
# Linux / macOS
./essential_tools/venv/bin/python essential_tools/verify.py

# Windows
essential_tools\venv\Scripts\python.exe essential_tools\verify.py
```

This checks the Python environment, every dependency, all 17 core modules, the API routes, your configuration, hardware, and the frontend files. Unconfigured optional backends produce warnings, not failures — running on mock is a legitimate state.

Expected output:

```
Python environment
  ✓ Python 3.12.3
  ✓ Running inside a venv (venv)

Dependencies
  ✓ fastapi — HTTP layer
  ✓ trimesh — mesh handling
  ...

ANVIL core
  ✓ anvil_core 1.0.0 — all 17 modules import
  ✓ HTTP server imports — 32 API routes registered
  ✓ 16 style presets loaded
  ✓ 10 pipeline steps defined

Ready to run.
```

### Full check — runs a real generation

```bash
./essential_tools/venv/bin/python essential_tools/verify.py --full
```

Adds an end-to-end pipeline run on mock backends:

```
Pipeline smoke test
  ✓ Full pipeline ran — 420 tris, 1260 verts
  ✓ Session checkpointed at 9/9 steps
```

If that passes, all nine generation steps, the style postprocess chain, and session checkpointing are working.

### Start the server

```bash
./essential_tools/run.sh          # Linux / macOS
run.bat                           # Windows
```

```
ANVIL starting on http://127.0.0.1:8765
```

### Verify the running server

```bash
curl http://127.0.0.1:8765/api/health
```

```json
{
  "ok": true,
  "version": "1.0.0",
  "gpu": { "cuda_available": true, "total_vram_gb": 12.0 },
  "blender": { "available": true },
  "backends": { "model": "mock", "image": "mock", "llm": "lm_studio (local-model)" }
}
```

Other useful endpoints:

```bash
curl http://127.0.0.1:8765/api/llm/health     # is the reasoning LLM reachable?
curl http://127.0.0.1:8765/api/styles         # all 16 style presets
curl http://127.0.0.1:8765/api/sessions       # resumable sessions
```

### Verify in the browser

Open **http://127.0.0.1:8765** and confirm:

1. **Topbar** shows your active backends with a green status dot.
2. **Setup panel** (left) shows three mode tabs and the style dropdown.
3. Choose **Direct text**, type something in *Object type*, press **Generate**.
4. The **centre viewport** shows pipeline steps ticking off with a live log.
5. A mesh appears, orbiting, with a triangle count in the corner.
6. **Activity rail** (right) — the Sessions tab now lists your generation.

Then try the batch queue: switch to **Batch queue**, add two or three items, press **Generate all**, and watch them run one at a time with per-row status.

---

## 6. Using the Blender addon

The addon runs the identical pipeline *inside* Blender, so scripts execute against the live scene instead of through a background subprocess.

1. Zip the `blender_addon/` folder.
2. In Blender: **Edit → Preferences → Add-ons → Install…**, select the zip, enable **"ANVIL — AI 3D asset pipeline"**.
3. If the addon lives outside the project directory, set an environment variable before launching Blender so it can find `anvil_core`:

```bash
export ANVIL_PROJECT_ROOT=/path/to/anvil
blender
```

4. The panel appears in the 3D viewport sidebar (press **N**) under the **ANVIL** tab.

The addon needs ANVIL's Python dependencies importable from Blender's Python. The simplest approach is to install them into Blender's bundled interpreter:

```bash
/path/to/blender/4.2/python/bin/python3.11 -m pip install trimesh numpy pillow httpx fast-simplification
```

---

## 7. Troubleshooting

**`python: command not found` / `'python' is not recognized`**
Python isn't installed or isn't on PATH. On Windows, reinstall and tick "Add Python to PATH". On macOS/Linux try `python3` explicitly.

**`No module named venv`**
Debian/Ubuntu splits it out: `sudo apt install python3-venv`.

**Setup finishes but `run.sh` says the environment is missing**
The venv didn't fully build. Delete and retry:
```bash
rm -rf essential_tools/venv && ./essential_tools/setup.sh
```

**`Can't reach LM Studio at http://127.0.0.1:1234/v1`**
LM Studio's server isn't running. Open LM Studio → Developer tab → Start Server, and confirm the port matches `LLM_ENDPOINT`.

**`OPENROUTER_API_KEY is empty`**
You set `LLM_PROVIDER=openrouter` without a key. Add it to `.env` and restart, or switch back to `lm_studio`.

**All styles produce identical triangle counts**
`fast-simplification` didn't install, so decimation silently fell back to leaving topology alone. Reinstall:
```bash
./essential_tools/venv/bin/pip install fast-simplification
```

**Mesh generates but never appears in Blender**
`BLENDER_PATH` isn't set. The UI warns about this on startup. The `.glb` is still in `sessions/<id>/final_mesh.glb`.

**Procedural mode fails immediately**
It needs Blender to execute generated `bpy` code. Set `BLENDER_PATH`, or switch procedural mode off.

**Viewport says "the 3D preview needs the bundled three.js files"**
Node wasn't available at setup time. Install Node 18+, then:
```bash
npm install && node essential_tools/vendor.js
```

**`Rig this` says UniRig isn't configured**
Working as intended — there's no mock rigger. Clone UniRig and set `UNIRIG_PATH`.

**Mesh is flagged `unremeshed`**
AutoRemesher isn't configured, so the remesh step was skipped rather than silently pretending the mesh has clean quad topology. Set `AUTOREMESHER_PATH`, or ignore it — the mesh is still usable.

**Port 8765 already in use**
Change `ANVIL_PORT` in `.env`.

**"A ... is already running" (409)**
Exactly one pipeline runs at a time by design — generation, batch, resume, edit, and rigging all share one gate, because they all contend for the same single VRAM slot. Wait for the current one to finish; the UI disables the affected controls while it's busy.

**Batch stopped partway through**
Expected after pressing *Stop after current*. Remaining items stay `pending` and the Sessions tab offers **Resume batch**.

**The generated mesh is flat — watertight, but paper-thin on one axis**
Image-to-3D reads the alpha channel to find the subject. A concept image with no clear subject margin gives it nothing to isolate, so it reconstructs a relief instead of a solid. Prompt for one object with space around it ("centered product photo, isolated on a plain background"), not a scene.

**"Texturing isn't set up yet — missing …"**
Working as intended: the **Texture** button names the missing prerequisite rather than failing after you press it. See § 4.4.

**The server process disappears during texturing**
`bpy` running in-process on a request worker. Blender's API needs the main thread and takes the process down rather than raising. ANVIL avoids this by passing `save_glb=False` and converting out-of-process — if you've edited [texturing.py](anvil_core/texturing.py) or `blender_bridge.py`, check both guards described in § 4.4 are intact.

**Everything slows to a crawl — generation taking 10× longer than it should**
The card is oversubscribed and Windows is paging VRAM to system RAM, which degrades instead of failing. Usually a local LLM holding VRAM while a 3D backend loads: at 12 GB there isn't room for both. ANVIL unloads LM Studio when it claims the slot, which needs LM Studio's **just-in-time loading** enabled so the model comes back on the next request. Check with `lms ps` mid-generation — it should report nothing loaded.

**Windows reports low memory though little seems to be running**
Check *commit charge*, not physical RAM: PyTorch's allocator reserves address space and returns none of it, so a long session can sit near the commit limit with a small working set. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set by default in [config.py](anvil_core/config.py)) mitigates it; restarting the server reclaims the rest immediately.

---

## Project layout

```
anvil/
├── anvil_core/              pipeline logic
│   ├── state_machine.py     the 10 steps
│   ├── batch.py             sequential queue runner
│   ├── edit_loop.py         property / shape / new_model / rig
│   ├── resource_manager.py  single-VRAM-resident policy
│   ├── model_gen.py         pluggable 3D backends
│   ├── postprocess.py       style-driven mesh chain
│   ├── texturing.py         PBR painting, post-generation
│   ├── status.py            pollable pipeline snapshot
│   └── ...
├── server/app.py            FastAPI + SSE
├── frontend/                three-zone workbench UI
├── blender_addon/           in-Blender panel
├── backends/                cloned generator repos (Hunyuan3D-2.1, …)
├── models/                  downloaded weights — hf/ · hy3dgen/ · u2net/
├── run.bat / run.sh         start the server
├── essential_tools/         setup.sh · setup.bat · verify.py
│   └── venv/                local Python environment
├── .env.example
└── requirements.txt
```
