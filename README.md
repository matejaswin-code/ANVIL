# ANVIL

**Local-first AI 3D asset generation pipeline**: a text prompt or reference image goes in, and a styled, production-ready 3D model comes out — delivered straight into your Blender scene.

Everything runs on your machine. Nothing installs globally.

```bash
# Windows
.\essential_tools\setup.bat
.\run.bat                        # then open http://127.0.0.1:8765

# Linux / macOS
./essential_tools/setup.sh
./essential_tools/run.sh         # then open http://127.0.0.1:8765
```

Out of the box, ANVIL runs on **mock backends** — the entire 10-step pipeline and web UI work on CPU with zero downloaded model weights, allowing you to verify your setup before downloading multi-gigabyte models.

👉 **Full setup instructions & GPU configuration: [SETUP_GUIDE.md](SETUP_GUIDE.md)**

---

## Key Features

- **Local-First & Complete Privacy**  
  All generation, reasoning, remeshing, and export operate on your local hardware. No forced telemetry, no cloud lock-in.

- **Intelligent Single-VRAM Model Juggler**  
  Built specifically for consumer GPUs (optimized for 12GB VRAM cards like the RTX 5070 / 4070 series and 32GB RAM). A strict, centralized `resource_manager` hot-swaps models sequentially — Reasoning LLM, Image Generator (SDXL / Diffusers), 3D Shape Engine (Hunyuan3D), and Rigger (UniRig) — preventing out-of-memory crashes.

- **Direct Two-Way Blender Bridge**  
  Generated meshes land directly inside your active `.blend` scene (embedded addon or background CLI). Continue editing your model conversationally in plain English (*"make it bigger"*, *"paint it rust red"*, *"rig this character"*, *"sharpen edges"*).

- **Three Generation Modes**  
  1. **Text → Image**: Interactive 3D-safety checklist ensuring watertight geometry and clean silhouettes, followed by concept image generation, then 3D synthesis.  
  2. **Direct Text**: Fast prompt-to-3D routing with automatic concept fallback.  
  3. **Image Upload**: Upload your own 2D concept or photo with automated background removal and canvas normalization.

- **16 Curated Style Presets**  
  From retro **PS1** and **Low-Poly** to **Stylized RPG**, **Hard Surface**, **Claymation**, and **Production Sculpt**. Presets couple prompt engineering modifiers with dedicated mesh post-processing pipelines (voxel remeshing, decimation, smoothing).

- **Crash-Resilient 10-Step State Machine**  
  Every generation step checkpoints `session.json` to disk. If interrupted, simply resume from the UI or browse past sessions without lost work.

- **On-Demand Texturing & Auto-Rigging**  
  Decoupled post-processing: only spend GPU time texturing or generating humanoid skeletons on models you approve.  
  > ⚠️ **Important Resource & Status Notes**:  
  > - **Skeleton Rigging**: The skeleton auto-rigging feature (UniRig integration) is **still untested**.  
  > - **RAM Requirements**: For **model generation only, 16GB RAM should be fine**. However, for **texture generation, 32GB RAM is the minimum** due to texture projection, baking, and rasterization overhead.

---

## LLM & API Key Support

ANVIL leverages Large Language Models for prompt enhancement, 3D checklist verification, and natural language Blender edits:

- **Local LLMs**: Out-of-the-box support for [LM Studio](https://lmstudio.ai/) (default `http://127.0.0.1:1234/v1`) using lightweight local reasoning models (Gemma 2, Qwen 2.5, Llama 3).
- **Google Gemini API**: Native cloud API support when you want zero local LLM VRAM overhead.
- **OpenRouter Support**: 🚀 **OpenRouter API key support will be added soon!** This will allow you to route prompt expansion and live scene edits through any model (Claude 3.7, DeepSeek R1, GPT-4o, etc.) with a single API key.

---

## ANVIL 1.1 Update Plan & Roadmap

The upcoming **ANVIL 1.1** release brings major advancements to shape fidelity and user control:

1. **Multi-View Shape Conditioning (`MULTIVIEW`)**  
   - Transition to `hunyuan3d-dit-v2-mv-turbo` for multi-angle shape generation.
   - Accept 1 to 4 directional camera inputs (`front`, `left`, `back`, `right`) for complete 360° geometric accuracy without occluded backsides.
   - Graceful single-image fallback (`{front: ...}`).

2. **Geometric Prior Conditioning (`CONTROLLED`)**  
   - Integration with geometric control signals:
     - **Bounding Box (BBOX)**: Enforce exact canonical dimensions, scale, and aspect ratios.
     - **Voxel Grid**: Remesh rough Blender blockouts into detailed assets.
     - **Point Clouds**: Condition on coarse 3D scans or depth projections.
     - **Skeletons**: Drive generation from Blender armature poses.

3. **View Preprocessing & Validation Gate**  
   - Headless image pipeline: EXIF rotation normalization, u2net ONNX background removal, subject bounding-box crop, and scale normalization (consistent 85% canvas fill).
   - Pre-generation validation gate: DINOv2 cross-view cosine similarity checks, scale consistency spread warnings, and near-duplicate slot detection.

4. **LM Studio VLM Quality Gate & Prompt Rewriter**  
   - Automated 3D-safety prompt rewriter eliminating backgrounds, complex lighting, and text artifacts.
   - Vision-language model evaluation gate to filter poor candidate concept images before generating meshes.

5. **Image Quality Stack**  
   - Best-of-N candidate sampling.
   - Supersampling and lighting delighting passes.
   - Framing LoRA support for SDXL / Diffusers backends.

6. **Material Controls & Blender Control Export**  
   - Pre-texture material property controls (metallic, roughness, surface flags).
   - One-click Blender control signal export (bounding boxes & armatures).
   - Built-in side-by-side A/B comparison harness.

---

## The Interface

A responsive, industrial three-zone workbench:

- **Left Rail — Setup**: Mode selection, style presets, checklist parameters, batch queue configuration.
- **Center — Viewport**: Embedded Three.js 3D viewport with real-time pipeline step progression, wireframe overlay, lighting controls, undo/redo, and conversational edit input.
- **Right Rail — Activity**: Live event logs, batch manager, and session browser for resuming past sessions.

---

## System Requirements
 
- **OS**: Windows 10/11 or Linux (Ubuntu 22.04+ recommended).
- **GPU**: NVIDIA GPU with 8GB+ VRAM (12GB+ recommended for full local Hunyuan3D + SDXL pipelines).
- **RAM**:
  - **Model Generation only**: **16GB RAM** should be fine.
  - **Texture Generation**: **32GB RAM minimum** is required for texture projection and map generation.
- **Skeleton Rigging Status**: The skeleton rigging feature is currently experimental and **still not tested**.
- **Python**: Python 3.10 – 3.13.
- **Blender**: Blender 4.x or 5.x (optional, for direct scene bridge and live editing).
- **Node.js**: Node.js 18+ (optional, for in-browser 3D preview bundling).

See [SETUP_GUIDE.md](SETUP_GUIDE.md) for full setup instructions, model weights installation, and environment configuration.
