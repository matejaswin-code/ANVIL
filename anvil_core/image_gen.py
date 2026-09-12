"""Image generation (pipeline step 5 — Text→Image mode only).

Pluggable the same way the 3D backends are (§2.5.1). The `mock` backend produces
a real PNG with no GPU at all, so the full 10-step pipeline is runnable and
testable on any machine before a single model weight is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

from . import events
from .config import SETTINGS
from .resource_manager import RESOURCE_MANAGER


class ImageGenError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# mock backend — deterministic, CPU-only, produces a genuine PNG
# ---------------------------------------------------------------------------

def _generate_mock(prompt: str, out_path: Path, size: int = 768) -> Path:
    from PIL import Image, ImageDraw

    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    rng_r = (seed & 0xFF)
    rng_g = ((seed >> 8) & 0xFF)
    rng_b = ((seed >> 16) & 0xFF)

    img = Image.new("RGB", (size, size), (18, 18, 20))
    draw = ImageDraw.Draw(img)

    # A deterministic abstract "concept image" derived from the prompt hash — it
    # stands in for a real diffusion result without pretending to be one.
    for i in range(12):
        s = seed >> (i * 3)
        x0 = (s % size)
        y0 = ((s >> 5) % size)
        w = 40 + ((s >> 9) % (size // 3))
        h = 40 + ((s >> 13) % (size // 3))
        col = ((rng_r + i * 11) % 256, (rng_g + i * 7) % 256, (rng_b + i * 13) % 256)
        draw.ellipse([x0, y0, x0 + w, y0 + h], outline=col, width=3)

    draw.rectangle([0, size - 60, size, size], fill=(10, 10, 11))
    draw.text((14, size - 44), "ANVIL mock concept image", fill=(221, 47, 41))
    draw.text((14, size - 28), prompt[:80], fill=(154, 151, 143))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# ComfyUI backend — talks to a running ComfyUI instance over HTTP
# ---------------------------------------------------------------------------

def _generate_comfyui(prompt: str, out_path: Path, session_id: str = "") -> Path:
    base = SETTINGS.comfyui_endpoint.rstrip("/")
    workflow_path = Path(__file__).parent / "workflows" / "txt2img.json"
    if not workflow_path.exists():
        raise ImageGenError(
            f"ComfyUI backend selected but no workflow found at {workflow_path}. "
            "Export an API-format workflow from ComfyUI and save it there."
        )
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    # Inject the prompt into every positive CLIPTextEncode node
    for node in workflow.values():
        if node.get("class_type") == "CLIPTextEncode" and node.get("_meta", {}).get("title", "").lower().startswith("positive"):
            node["inputs"]["text"] = prompt

    try:
        with httpx.Client(timeout=300.0) as client:
            res = client.post(f"{base}/prompt", json={"prompt": workflow})
            if res.status_code >= 400:
                raise ImageGenError(f"ComfyUI returned HTTP {res.status_code}: {res.text[:200]}")
            prompt_id = res.json().get("prompt_id")
            if not prompt_id:
                raise ImageGenError("ComfyUI accepted the job but returned no prompt_id")

            # Poll history until the job lands
            deadline = time.time() + 300
            while time.time() < deadline:
                hist = client.get(f"{base}/history/{prompt_id}")
                data = hist.json() if hist.status_code == 200 else {}
                if prompt_id in data:
                    outputs = data[prompt_id].get("outputs", {})
                    for node_out in outputs.values():
                        for image in node_out.get("images", []):
                            img = client.get(f"{base}/view", params={
                                "filename": image["filename"],
                                "subfolder": image.get("subfolder", ""),
                                "type": image.get("type", "output"),
                            })
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            out_path.write_bytes(img.content)
                            return out_path
                    raise ImageGenError("ComfyUI job finished but produced no image output")
                time.sleep(1.5)
            raise ImageGenError("ComfyUI job timed out after 300s")
    except httpx.ConnectError as exc:
        raise ImageGenError(
            f"Can't reach ComfyUI at {base}. Is it running? Check COMFYUI_ENDPOINT in .env."
        ) from exc


# ---------------------------------------------------------------------------
# diffusers backend — local, occupies the VRAM slot
# ---------------------------------------------------------------------------

class _DiffusersPipeline:
    def __init__(self, model_id: str):
        from diffusers import AutoPipelineForText2Image  # noqa: PLC0415
        import torch  # noqa: PLC0415
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            model_id, torch_dtype=torch.float16, variant="fp16"
        ).to("cuda")

    def __call__(self, prompt: str, out_path: Path) -> Path:
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=SETTINGS.image_steps,
            guidance_scale=SETTINGS.image_guidance,
            width=SETTINGS.image_size,
            height=SETTINGS.image_size,
        ).images[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return out_path

    def unload(self) -> None:
        del self.pipe


def _generate_diffusers(prompt: str, out_path: Path, session_id: str = "") -> Path:
    model_id = SETTINGS.image_model_id
    pipe = RESOURCE_MANAGER.ensure_loaded(
        kind="image_gen",
        key=f"diffusers:{model_id}",
        loader=lambda: _DiffusersPipeline(model_id),
        unloader=lambda p: p.unload(),
        session_id=session_id,
    )
    return pipe(prompt, out_path)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def generate_image(prompt: str, out_path: Path, *, session_id: str = "") -> Path:
    backend = SETTINGS.image_backend
    events.log(session_id, f"Generating concept image via '{backend}' backend")

    if backend == "mock":
        # Mock claims no VRAM slot — it's CPU-only by design.
        return _generate_mock(prompt, out_path)
    if backend == "comfyui":
        # ComfyUI runs out-of-process and manages its own VRAM; we eject ours so
        # the two don't fight over the card (§2.5.5).
        RESOURCE_MANAGER.eject(session_id=session_id)
        return _generate_comfyui(prompt, out_path, session_id=session_id)
    if backend == "diffusers":
        return _generate_diffusers(prompt, out_path, session_id=session_id)

    raise ImageGenError(
        f"Unknown IMAGE_BACKEND {backend!r}. Valid options: mock, comfyui, diffusers."
    )
