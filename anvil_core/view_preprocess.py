"""View image preprocessing.

Identical pipeline for every view, in both MULTIVIEW and CONTROLLED modes.
Divergence between views is the main multi-view failure mode, so the same
operations run in the same order with the same thresholds on every slot.

Pipeline per view:
    1. EXIF-normalize on load — rotation metadata silently flips side views.
    2. Background removal — same model, same threshold, every view.
    3. Crop to object bbox from alpha, pad square, center.
    4. Scale-normalize so the object fills ~0.85 of the canvas.
    5. Resize to the model's expected input resolution.

This module is headless and CLI-testable — no server, no GPU models (u2net runs
on CPU via ONNX), no Blender dependency.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from . import events
from .genrequest import GenRequest, View


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

# Target fraction of the canvas the object should fill after normalization.
TARGET_FILL = 0.85

# Model input resolution — Hunyuan3D-2mv expects 512×512 RGBA.
MODEL_INPUT_SIZE = 512

# Minimum alpha coverage (fraction of image area) to consider matting valid.
MIN_ALPHA_COVERAGE = 0.01

# Maximum alpha coverage — above this, the image is probably un-matted.
MAX_ALPHA_COVERAGE = 0.95


@dataclass
class PreprocessResult:
    """Per-view result from the preprocessing pipeline."""
    view: View
    original_size: tuple[int, int]
    bbox: tuple[int, int, int, int]       # x0, y0, x1, y1 of object in original
    scale_ratio: float                     # object_bbox_area / canvas_area
    alpha_coverage: float                  # fraction of non-zero alpha pixels
    warnings: list[str]


# ---------------------------------------------------------------------------
# 1. EXIF normalize
# ---------------------------------------------------------------------------

def _exif_normalize(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation tag and strip metadata.

    Rotation metadata silently flips side views — a Left image that displays
    correctly in a viewer may arrive physically rotated, and the model sees
    the physical pixels, not the EXIF hint.
    """
    return ImageOps.exif_transpose(img)


# ---------------------------------------------------------------------------
# 2. Background removal
# ---------------------------------------------------------------------------

_BG_REMOVER = None


def _get_bg_remover():
    """Lazy-load the u2net ONNX background remover.

    Uses the u2net.onnx weights from models/u2net/ (same as Hunyuan3D's
    hy3dshape.rembg but loaded independently so preprocessing doesn't
    require the full Hunyuan3D repo).
    """
    global _BG_REMOVER
    if _BG_REMOVER is not None:
        return _BG_REMOVER

    from .config import MODELS_DIR  # noqa: PLC0415

    onnx_path = MODELS_DIR / "u2net" / "u2net.onnx"
    if not onnx_path.exists():
        # Fall back: try importing rembg which can auto-download
        try:
            import rembg  # noqa: PLC0415, F401
            _BG_REMOVER = "rembg"
            return _BG_REMOVER
        except ImportError:
            pass
        return None

    try:
        import onnxruntime as ort  # noqa: PLC0415
        _BG_REMOVER = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
    except ImportError:
        # onnxruntime not installed — try rembg
        try:
            import rembg  # noqa: PLC0415, F401
            _BG_REMOVER = "rembg"
        except ImportError:
            _BG_REMOVER = None

    return _BG_REMOVER


def _u2net_predict(session, img_rgb: np.ndarray) -> np.ndarray:
    """Run u2net ONNX to get an alpha mask."""
    h, w = img_rgb.shape[:2]
    # u2net expects 320×320 RGB normalized to [0,1]
    resized = cv2.resize(img_rgb, (320, 320), interpolation=cv2.INTER_LINEAR)
    inp = resized.astype(np.float32) / 255.0
    # Normalize per-channel (ImageNet-style)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)[np.newaxis]  # NCHW

    outputs = session.run(None, {session.get_inputs()[0].name: inp})
    mask = outputs[0][0, 0]  # first output, first batch, first channel
    # Normalize to [0, 255]
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    mask = (mask * 255).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
    # Threshold to binary-ish alpha
    _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    return mask


def _remove_background(img: Image.Image) -> Image.Image:
    """Return an RGBA image with background removed via alpha channel.

    If the image already has meaningful alpha (min < 255), it's returned as-is.
    """
    # Check if alpha already carries information
    if img.mode == "RGBA":
        alpha = np.asarray(img)[..., 3]
        if alpha.min() < 250:  # has real transparency
            return img

    remover = _get_bg_remover()
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)

    if remover is None:
        # No background removal available — add full-white alpha and warn
        rgba = img.convert("RGBA")
        return rgba

    if remover == "rembg":
        import rembg  # noqa: PLC0415
        # rembg.remove returns RGBA PIL image
        result = rembg.remove(rgb)
        return result

    # ONNX u2net session
    mask = _u2net_predict(remover, arr)

    # Combine RGB + alpha
    rgba = np.dstack([arr, mask])
    return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------------------
# 3. Crop to bbox, pad square, center
# ---------------------------------------------------------------------------

def _crop_pad_center(img: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """Crop to the object's alpha bounding box, then pad to a square and center.

    Returns (processed_image, (x0, y0, x1, y1) bbox in original coords).
    """
    assert img.mode == "RGBA"
    alpha = np.asarray(img)[..., 3]

    # Find bounding box from alpha
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any() or not cols.any():
        # Empty alpha — return original with trivial bbox
        return img, (0, 0, img.width, img.height)

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    y1 += 1
    x1 += 1
    bbox = (int(x0), int(y0), int(x1), int(y1))

    # Crop to bbox
    cropped = img.crop(bbox)

    # Pad to square
    w, h = cropped.size
    side = max(w, h)
    padded = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    paste_x = (side - w) // 2
    paste_y = (side - h) // 2
    padded.paste(cropped, (paste_x, paste_y))

    return padded, bbox


# ---------------------------------------------------------------------------
# 4. Scale-normalize
# ---------------------------------------------------------------------------

def _scale_normalize(img: Image.Image, target_fill: float = TARGET_FILL) -> tuple[Image.Image, float]:
    """Scale the object so it fills target_fill fraction of the canvas.

    Returns (scaled_image, scale_ratio) where scale_ratio is the object's
    bbox area relative to the total canvas area.
    """
    assert img.mode == "RGBA"
    alpha = np.asarray(img)[..., 3]
    canvas_area = alpha.shape[0] * alpha.shape[1]

    # Current fill: fraction of canvas covered by non-zero alpha
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any():
        return img, 0.0

    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    obj_w = x1 - x0 + 1
    obj_h = y1 - y0 + 1

    # Current fill ratio (max dimension / canvas side)
    current_fill = max(obj_w, obj_h) / max(img.width, img.height)
    scale_ratio = (obj_w * obj_h) / canvas_area

    if current_fill < 0.01:
        return img, scale_ratio

    # Scale factor to reach target fill
    scale_factor = target_fill / current_fill

    if abs(scale_factor - 1.0) < 0.05:
        # Close enough — skip resampling to avoid quality loss
        return img, scale_ratio

    new_w = int(img.width * scale_factor)
    new_h = int(img.height * scale_factor)
    side = max(new_w, new_h)

    # Resize the object
    resized = img.resize((new_w, new_h), Image.LANCZOS)

    # Re-center on a square canvas
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    paste_x = (side - new_w) // 2
    paste_y = (side - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # Recalculate scale ratio
    alpha2 = np.asarray(canvas)[..., 3]
    rows2 = np.any(alpha2 > 0, axis=1)
    cols2 = np.any(alpha2 > 0, axis=0)
    if rows2.any():
        oy0, oy1 = np.where(rows2)[0][[0, -1]]
        ox0, ox1 = np.where(cols2)[0][[0, -1]]
        scale_ratio = ((ox1 - ox0 + 1) * (oy1 - oy0 + 1)) / (side * side)

    return canvas, scale_ratio


# ---------------------------------------------------------------------------
# 5. Resize to model input
# ---------------------------------------------------------------------------

def _resize_to_model_input(img: Image.Image, size: int = MODEL_INPUT_SIZE) -> Image.Image:
    """Resize to the model's expected input resolution."""
    if img.width == size and img.height == size:
        return img
    return img.resize((size, size), Image.LANCZOS)


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------

def preprocess_view(view: View, *, session_id: str = "") -> PreprocessResult:
    """Run the full 5-step preprocessing pipeline on a single view.

    Updates `view.image` in-place with the processed RGBA PIL image, and
    sets `view.scale_ratio`.
    """
    warnings: list[str] = []
    slot_name = view.slot.value

    # Load
    if not view.source_path.exists():
        raise FileNotFoundError(f"View {slot_name}: source image not found at {view.source_path}")

    img = Image.open(view.source_path)
    original_size = img.size

    # Step 1: EXIF normalize
    img = _exif_normalize(img)

    # Step 2: Background removal
    img = _remove_background(img)

    # Check alpha quality
    alpha = np.asarray(img)[..., 3]
    alpha_coverage = float(np.count_nonzero(alpha) / alpha.size) if alpha.size > 0 else 0.0

    if alpha_coverage < MIN_ALPHA_COVERAGE:
        warnings.append(f"{slot_name}: very low alpha coverage ({alpha_coverage:.1%}) — "
                        "matting may have failed or the image is mostly transparent")

    if alpha_coverage > MAX_ALPHA_COVERAGE:
        warnings.append(f"{slot_name}: very high alpha coverage ({alpha_coverage:.1%}) — "
                        "background removal may not have worked")

    # Step 3: Crop, pad square, center
    img, bbox = _crop_pad_center(img)

    # Step 4: Scale-normalize
    img, scale_ratio = _scale_normalize(img)

    # Step 5: Resize to model input
    img = _resize_to_model_input(img)

    # Update view in-place
    view.image = img
    view.scale_ratio = scale_ratio

    if session_id:
        events.log(session_id, f"Preprocessed {slot_name}: {original_size} → {img.size}, "
                   f"fill={scale_ratio:.2f}, alpha={alpha_coverage:.1%}")

    return PreprocessResult(
        view=view,
        original_size=original_size,
        bbox=bbox,
        scale_ratio=scale_ratio,
        alpha_coverage=alpha_coverage,
        warnings=warnings,
    )


def preprocess_request(request: GenRequest, *, session_id: str = "",
                       views_dir: Path | None = None) -> list[PreprocessResult]:
    """Preprocess all views in a GenRequest.

    Saves processed images to views_dir/<slot>.png if views_dir is provided.
    Returns a list of PreprocessResult, one per view.
    """
    results = []

    for slot, view in request.views.items():
        result = preprocess_view(view, session_id=session_id)
        results.append(result)

        # Save processed view to disk for checkpointing/debugging
        if views_dir and view.image is not None:
            views_dir.mkdir(parents=True, exist_ok=True)
            out_path = views_dir / f"{slot.value}.png"
            view.image.save(out_path)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Headless CLI for testing preprocessing on individual images."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="ANVIL view preprocessor")
    parser.add_argument("images", nargs="+", help="Image files to preprocess")
    parser.add_argument("-o", "--output", default=".", help="Output directory")
    parser.add_argument("--size", type=int, default=MODEL_INPUT_SIZE, help="Output size")
    parser.add_argument("--fill", type=float, default=TARGET_FILL, help="Target fill fraction")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in args.images:
        p = Path(img_path)
        if not p.exists():
            print(f"  SKIP  {p} — not found")
            continue

        from .genrequest import Slot  # noqa: PLC0415
        view = View(slot=Slot.FRONT, source_path=p)

        result = preprocess_view(view)
        out_file = out_dir / f"{p.stem}_preprocessed.png"

        # Resize to requested size if different from default
        if args.size != MODEL_INPUT_SIZE and view.image is not None:
            view.image = _resize_to_model_input(view.image, size=args.size)

        view.image.save(out_file)

        print(f"  OK    {p.name} → {out_file.name}")
        print(f"        original={result.original_size} bbox={result.bbox}")
        print(f"        scale_ratio={result.scale_ratio:.3f} alpha={result.alpha_coverage:.1%}")
        for w in result.warnings:
            print(f"        WARN: {w}")


if __name__ == "__main__":
    main()
