"""3D generation backends — the `Generator` plugin interface (§2.5.1).

All backends implement the same three-method contract, so swapping one for
another is a Settings dropdown change and nothing else:

    load()      — bring weights into VRAM (claims the single resident slot)
    generate()  — image-or-text in, raw .glb path out
    unload()    — free VRAM

The `mock` backend implements the contract with trimesh on the CPU, producing a
genuinely valid .glb. That makes the whole pipeline verifiable end to end on a
machine with no GPU and no downloaded weights — which is what Milestone 1's
acceptance test actually needs to be checkable.

The `request` parameter carries the validated, slot-keyed view/control payload
(`GenRequest | None`). Generators accept it but are not required to read it yet
(the MULTIVIEW front-only path is still image_path).
"""
from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

from . import events
from .config import SETTINGS
from .resource_manager import RESOURCE_MANAGER


class ModelGenError(RuntimeError):
    pass


class Generator(ABC):
    """The plugin contract. A new backend is a subclass plus a REGISTRY entry."""

    name: str = "abstract"
    vram_estimate_gb: float = 0.0
    accepts: tuple[str, ...] = ("image", "text")

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, *, prompt: str, image_path: str | None, out_path: Path, style, flags: dict,
                 request=None) -> Path: ...

    @abstractmethod
    def unload(self) -> None: ...


# ---------------------------------------------------------------------------
# mock backend — CPU-only, produces a real .glb
# ---------------------------------------------------------------------------

class MockGenerator(Generator):
    name = "mock"
    vram_estimate_gb = 0.0

    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def generate(self, *, prompt: str, image_path: str | None, out_path: Path, style, flags: dict,
                 request=None) -> Path:
        import numpy as np
        import trimesh

        seed = int(hashlib.sha256((prompt or "anvil").encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        # Build a deterministic blocky "prop" from the prompt hash — a union of a
        # few primitives, so the result is a valid watertight-ish mesh with real
        # topology for postprocess to chew on rather than a placeholder cube.
        parts = []
        base = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
        parts.append(base)

        n_parts = 3 + (seed % 4)
        for i in range(n_parts):
            s = seed >> (i * 5)
            kind = s % 3
            scale = 0.25 + (s % 60) / 100.0
            offset = (rng.random(3) - 0.5) * 1.2
            if kind == 0:
                part = trimesh.creation.box(extents=(scale, scale * 1.4, scale * 0.7))
            elif kind == 1:
                part = trimesh.creation.icosphere(subdivisions=2, radius=scale * 0.6)
            else:
                part = trimesh.creation.cylinder(radius=scale * 0.4, height=scale * 1.6, sections=16)
            part.apply_translation(offset)
            parts.append(part)

        mesh = trimesh.util.concatenate(parts)
        mesh.apply_scale(1.0 / max(mesh.extents))   # normalise to roughly unit size

        # Give it a deterministic base colour so style changes are visible downstream
        color = [(seed & 0xFF), ((seed >> 8) & 0xFF), ((seed >> 16) & 0xFF), 255]
        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, face_colors=color)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(out_path)
        return out_path


# ---------------------------------------------------------------------------
# TRELLIS.2 backends
# ---------------------------------------------------------------------------

class Trellis2Generator(Generator):
    """TRELLIS.2, both GGUF-quantised and full-precision variants."""

    def __init__(self, quantised: bool = True):
        self.quantised = quantised
        self.name = "trellis2_gguf" if quantised else "trellis2_full"
        self.vram_estimate_gb = 6.0 if quantised else 11.0
        self._pipeline = None

    def load(self) -> None:
        repo = SETTINGS.trellis2_path
        if not repo or not Path(repo).exists():
            raise ModelGenError(
                f"TRELLIS.2 selected but TRELLIS2_PATH is unset or missing ({repo!r}). "
                "Clone the TRELLIS repo, point TRELLIS2_PATH at it in .env, and re-run. "
                "Until then, set MODEL_BACKEND=mock to exercise the rest of the pipeline."
            )
        import sys
        if repo not in sys.path:
            sys.path.insert(0, repo)
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ModelGenError(
                f"TRELLIS.2 is at {repo} but its Python package isn't importable ({exc}). "
                "Did its own install step run inside this venv?"
            ) from exc
        variant = "JeffreyXiang/TRELLIS-image-large"
        self._pipeline = TrellisImageTo3DPipeline.from_pretrained(variant)
        self._pipeline.cuda()

    def unload(self) -> None:
        self._pipeline = None

    def generate(self, *, prompt: str, image_path: str | None, out_path: Path, style, flags: dict,
                 request=None) -> Path:
        if self._pipeline is None:
            raise ModelGenError("TRELLIS.2 generate() called before load()")
        if not image_path:
            raise ModelGenError("TRELLIS.2 is an image-conditioned model — step 6 should have supplied an image.")
        from PIL import Image
        import trimesh

        image = Image.open(image_path).convert("RGB")
        result = self._pipeline.run(
            image,
            seed=1,
            formats=["mesh"],
            sparse_structure_sampler_params={"steps": 12},
            slat_sampler_params={"steps": 12},
        )
        mesh = result["mesh"][0]
        tm = trimesh.Trimesh(vertices=mesh.vertices.cpu().numpy(), faces=mesh.faces.cpu().numpy())
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tm.export(out_path)
        return out_path


class Hunyuan3DGenerator(Generator):
    name = "hunyuan3d"
    vram_estimate_gb = 9.0

    def __init__(self):
        self._pipeline = None

    def load(self) -> None:
        repo = SETTINGS.hunyuan3d_path
        if not repo or not Path(repo).exists():
            raise ModelGenError(
                f"Hunyuan3D selected but HUNYUAN3D_PATH is unset or missing ({repo!r}). "
                "Clone Hunyuan3D-2.1, point HUNYUAN3D_PATH at it in .env, and re-run. "
                "Until then, set MODEL_BACKEND=mock."
            )
        import sys
        # 2.1 splits the repo into hy3dshape/ (geometry) and hy3dpaint/ (texture); the
        # importable package sits one level down. Only the shape half is used here —
        # hy3dpaint needs compiled CUDA rasterisers, and ANVIL does its own texturing.
        pkg_root = Path(repo) / "hy3dshape"
        entry = str(pkg_root if pkg_root.exists() else repo)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ModelGenError(
                f"Hunyuan3D is at {repo} but its package isn't importable ({exc}). "
                "Expected the 2.1 layout with a hy3dshape/ subdirectory."
            ) from exc
        # Defaults already resolve to subfolder='hunyuan3d-dit-v2-1', variant='fp16'.
        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")

    def unload(self) -> None:
        # Drop the module references too — the pipeline holds VAE/DiT/conditioner
        # submodules, and leaving them bound would keep the weights on the card after
        # the resource manager has already declared the slot free.
        pipe, self._pipeline = self._pipeline, None
        if pipe is not None:
            for attr in ("vae", "model", "conditioner"):
                if hasattr(pipe, attr):
                    setattr(pipe, attr, None)
        del pipe

    @staticmethod
    def _isolate_subject(image):
        """Return an RGBA image whose alpha marks the subject.

        Hunyuan3D reads the alpha channel to find the object. A concept image straight
        from the image backend is opaque edge to edge, so the model treats the whole
        frame as the subject and returns a flat slab instead of a solid. Only run the
        background remover when the alpha carries no information — an image that
        already has a real cutout (a user upload, say) is left alone.
        """
        import numpy as np  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        if image.mode == "RGBA" and np.asarray(image)[..., 3].min() < 255:
            return image
        from hy3dshape.rembg import BackgroundRemover  # noqa: PLC0415
        return BackgroundRemover()(image.convert("RGB"))

    def generate(self, *, prompt: str, image_path: str | None, out_path: Path, style, flags: dict,
                 request=None) -> Path:
        if self._pipeline is None:
            raise ModelGenError("Hunyuan3D generate() called before load()")
        if not image_path:
            raise ModelGenError("Hunyuan3D is image-conditioned — step 6 should have supplied an image.")
        from PIL import Image
        image = Image.open(image_path)
        image = self._isolate_subject(image)
        # mc_algo='mc' picks skimage's marching cubes. The alternative, 'dmc', needs the
        # `diso` CUDA extension, which has no prebuilt Windows wheel.
        # generate() has no session_id in its signature; the dispatcher already logged
        # which backend is running, so this rides along unattributed.
        events.log(
            "",
            f"Hunyuan3D: octree={SETTINGS.mesh_octree_resolution} "
            f"steps={SETTINGS.mesh_inference_steps} guidance={SETTINGS.mesh_guidance}",
        )
        result = self._pipeline(
            image=image,
            mc_algo="mc",
            octree_resolution=SETTINGS.mesh_octree_resolution,
            num_inference_steps=SETTINGS.mesh_inference_steps,
            guidance_scale=SETTINGS.mesh_guidance,
        )[0]
        # 2.1 is typed List[List[Trimesh]] but returns a bare mesh for a single input.
        mesh = result[0] if isinstance(result, (list, tuple)) else result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_path))
        return out_path


class Hunyuan3DMvGenerator(Generator):
    """Multi-view shape generation via Hunyuan3D-2mv (dit-v2-mv-turbo).

    Uses the FlowMatchingPipeline from the same hy3dshape package as the base
    Hunyuan3D backend, but loads the 2mv-turbo checkpoint which accepts a
    slot-keyed dict: ``{"front": img, "left": img, ...}``.

    Only ``front`` is mandatory — omitting other slots degrades gracefully.
    A single-image run is simply ``{"front": img}``, which is the n=1 parity
    test target.
    """
    name = "hunyuan3d_mv"
    vram_estimate_gb = 6.0  # turbo distillation is lighter than base

    def __init__(self):
        self._pipeline = None

    def load(self) -> None:
        repo = SETTINGS.hunyuan3d_path
        if not repo or not Path(repo).exists():
            raise ModelGenError(
                f"Hunyuan3D MV selected but HUNYUAN3D_PATH is unset or missing ({repo!r}). "
                "Clone Hunyuan3D-2.1 (which contains the hy3dshape package), point "
                "HUNYUAN3D_PATH at it in .env, and re-run."
            )
        import sys
        pkg_root = Path(repo) / "hy3dshape"
        entry = str(pkg_root if pkg_root.exists() else repo)
        if entry not in sys.path:
            sys.path.insert(0, entry)
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ModelGenError(
                f"Hunyuan3D is at {repo} but its package isn't importable ({exc}). "
                "Expected the 2.1 layout with a hy3dshape/ subdirectory."
            ) from exc

        # Load the multi-view turbo checkpoint. The subfolder is
        # "hunyuan3d-dit-v2-mv-turbo" under the tencent/Hunyuan3D-2mv HF repo.
        # If the user has downloaded it locally, HUNYUAN3D_MV_MODEL can override.
        mv_model = SETTINGS.hunyuan3d_mv_model or "tencent/Hunyuan3D-2mv"
        mv_subfolder = SETTINGS.hunyuan3d_mv_subfolder or "hunyuan3d-dit-v2-mv-turbo"

        events.log("", f"Loading Hunyuan3D-MV: model={mv_model} subfolder={mv_subfolder}")
        self._pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            mv_model,
            subfolder=mv_subfolder,
            use_safetensors=True,
        )

    def unload(self) -> None:
        pipe, self._pipeline = self._pipeline, None
        if pipe is not None:
            for attr in ("vae", "model", "conditioner"):
                if hasattr(pipe, attr):
                    setattr(pipe, attr, None)
        del pipe

    @staticmethod
    def _isolate_subject(image):
        """Return an RGBA image with subject isolated (same as base Hunyuan3D)."""
        import numpy as np  # noqa: PLC0415
        if image.mode == "RGBA" and np.asarray(image)[..., 3].min() < 255:
            return image
        try:
            from hy3dshape.rembg import BackgroundRemover  # noqa: PLC0415
            return BackgroundRemover()(image.convert("RGB"))
        except ImportError:
            # If rembg isn't available, return as RGBA with full alpha
            return image.convert("RGBA")

    def _build_view_dict(self, image_path: str | None, request) -> dict:
        """Build the slot-keyed view dict from request or fall back to front-only."""
        from PIL import Image as PILImage  # noqa: PLC0415

        view_dict = {}

        if request is not None and hasattr(request, 'views') and request.views:
            # Use preprocessed views from the GenRequest
            for slot, view in request.views.items():
                if view.image is not None:
                    view_dict[slot.value] = self._isolate_subject(view.image)
                elif view.source_path and view.source_path.exists():
                    img = PILImage.open(view.source_path)
                    view_dict[slot.value] = self._isolate_subject(img)

        if not view_dict and image_path:
            # Fall back to single front image
            img = PILImage.open(image_path)
            view_dict["front"] = self._isolate_subject(img)

        if not view_dict:
            raise ModelGenError(
                "Hunyuan3D-MV needs at least a front image — "
                "neither the request nor image_path provided one."
            )

        return view_dict

    def generate(self, *, prompt: str, image_path: str | None, out_path: Path, style, flags: dict,
                 request=None) -> Path:
        if self._pipeline is None:
            raise ModelGenError("Hunyuan3D-MV generate() called before load()")

        view_dict = self._build_view_dict(image_path, request)

        slot_list = ", ".join(view_dict.keys())
        events.log(
            "",
            f"Hunyuan3D-MV: slots=[{slot_list}] "
            f"octree={SETTINGS.mesh_octree_resolution} "
            f"steps={SETTINGS.mesh_inference_steps} "
            f"guidance={SETTINGS.mesh_guidance}",
        )

        result = self._pipeline(
            image=view_dict,
            mc_algo="mc",
            octree_resolution=SETTINGS.mesh_octree_resolution,
            num_inference_steps=SETTINGS.mesh_inference_steps,
            guidance_scale=SETTINGS.mesh_guidance,
        )[0]

        mesh = result[0] if isinstance(result, (list, tuple)) else result
        out_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(str(out_path))
        return out_path


# ---------------------------------------------------------------------------
# registry + dispatch
# ---------------------------------------------------------------------------

REGISTRY: dict[str, callable] = {
    "mock": lambda: MockGenerator(),
    "trellis2_gguf": lambda: Trellis2Generator(quantised=True),
    "trellis2_full": lambda: Trellis2Generator(quantised=False),
    "hunyuan3d": lambda: Hunyuan3DGenerator(),
    "hunyuan3d_mv": lambda: Hunyuan3DMvGenerator(),
}

BACKEND_INFO = [
    {"id": "mock", "label": "Mock (CPU, no weights)", "vram_gb": 0.0},
    {"id": "trellis2_gguf", "label": "TRELLIS.2 GGUF (~6GB VRAM)", "vram_gb": 6.0},
    {"id": "trellis2_full", "label": "TRELLIS.2 full-precision (~11GB)", "vram_gb": 11.0},
    {"id": "hunyuan3d", "label": "Hunyuan3D 2.1 (~9GB)", "vram_gb": 9.0},
    {"id": "hunyuan3d_mv", "label": "Hunyuan3D 2mv-turbo (~6GB, multi-view)", "vram_gb": 6.0},
]


def _build(backend_id: str) -> Generator:
    if backend_id not in REGISTRY:
        raise ModelGenError(
            f"Unknown MODEL_BACKEND {backend_id!r}. Valid options: {', '.join(REGISTRY)}."
        )
    return REGISTRY[backend_id]()


def generate_mesh(
    *, prompt: str, image_path: str | None, out_path: Path, style, flags: dict,
    session_id: str = "", request=None,
) -> Path:
    """Step 7. Claims the VRAM slot, generates, leaves the backend resident
    (the next generation almost certainly wants the same one).

    `request` is the validated GenRequest sidecar. Generators accept it
    but are not required to consume it until multi-view routing is enabled.
    """
    backend_id = SETTINGS.model_backend
    events.log(session_id, f"3D generation via '{backend_id}' backend")

    if backend_id == "mock":
        return _build("mock").generate(
            prompt=prompt, image_path=image_path, out_path=out_path, style=style, flags=flags,
            request=request,
        )

    def _loader() -> Generator:
        gen = _build(backend_id)
        gen.load()
        return gen

    generator: Generator = RESOURCE_MANAGER.ensure_loaded(
        kind="model_gen",
        key=backend_id,
        loader=_loader,
        unloader=lambda g: g.unload(),
        session_id=session_id,
    )
    mesh_path = generator.generate(
        prompt=prompt, image_path=image_path, out_path=out_path, style=style, flags=flags,
        request=request,
    )

    # Guard against the degenerate-mesh case before anything downstream trusts it (§9)
    _assert_non_degenerate(mesh_path)
    return mesh_path


def _assert_non_degenerate(path: Path) -> None:
    try:
        import trimesh
        mesh = trimesh.load(path, force="mesh")
        if mesh is None or len(mesh.vertices) == 0 or len(mesh.faces) == 0:
            raise ModelGenError(
                "3D backend returned a degenerate mesh (zero vertices or faces). "
                "Nothing was imported — try regenerating, or a different prompt/backend."
            )
    except ModelGenError:
        raise
    except Exception as exc:
        raise ModelGenError(f"Generated mesh at {path} couldn't be read back: {exc}") from exc
