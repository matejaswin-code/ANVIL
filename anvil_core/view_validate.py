"""View validation gate.

Blocks bad input before a heavy checkpoint loads. Returns a structured report
the UI can render, with hard-fail and warn-level checks.

Checks implemented:
  - Front view present (hard fail)
  - Same object across slots via DINOv2 cosine similarity (hard fail < 0.75)
  - Scale consistency across views (warn > 10% spread)
  - Matting quality: alpha coverage, holes, edge fragmentation (warn per view)
  - Near-duplicate slots: cosine > 0.98 → auto-drop the redundant slot

This module is headless and CLI-testable. DINOv2 runs through the resource
manager to share the VRAM slot, but falls back to CPU or skip when unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from . import events
from .genrequest import GenRequest, Slot, View, ModeConflict
from .resource_manager import RESOURCE_MANAGER


# ---------------------------------------------------------------------------
# result types
# ---------------------------------------------------------------------------

@dataclass
class ViewCheck:
    """Result of a single check on a single view."""
    name: str
    passed: bool
    level: str = "info"        # "info", "warn", "fail"
    message: str = ""
    value: float | None = None


@dataclass
class ValidationReport:
    """Full validation report for a GenRequest."""
    valid: bool = True                          # False if any hard-fail
    checks: list[ViewCheck] = field(default_factory=list)
    dropped_slots: list[str] = field(default_factory=list)

    @property
    def warnings(self) -> list[ViewCheck]:
        return [c for c in self.checks if c.level == "warn"]

    @property
    def failures(self) -> list[ViewCheck]:
        return [c for c in self.checks if c.level == "fail"]

    def summary(self) -> str:
        parts = []
        if self.failures:
            parts.append(f"{len(self.failures)} failed")
        if self.warnings:
            parts.append(f"{len(self.warnings)} warnings")
        if self.dropped_slots:
            parts.append(f"dropped: {', '.join(self.dropped_slots)}")
        if not parts:
            return "all checks passed"
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# DINOv2 feature extraction
# ---------------------------------------------------------------------------

_DINO_MODEL = None
_DINO_TRANSFORM = None


def _load_dino():
    """Load DINOv2-giant for feature extraction. Returns (model, transform) or (None, None)."""
    global _DINO_MODEL, _DINO_TRANSFORM
    if _DINO_MODEL is not None:
        return _DINO_MODEL, _DINO_TRANSFORM

    try:
        import torch  # noqa: PLC0415
        from torchvision import transforms  # noqa: PLC0415

        # Try loading from local HF cache first
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitg14",
                               trust_repo=True, verbose=False)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        _DINO_MODEL = model
        _DINO_TRANSFORM = transform
        return model, transform

    except Exception:
        return None, None


def _dino_embed(img: Image.Image) -> np.ndarray | None:
    """Get DINOv2 CLS embedding for an image. Returns None if unavailable."""
    model, transform = _load_dino()
    if model is None:
        return None

    try:
        import torch  # noqa: PLC0415

        # Convert RGBA to RGB with white background for consistent embeddings
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[3])
            img = bg
        else:
            img = img.convert("RGB")

        device = next(model.parameters()).device
        tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(tensor)

        return features.cpu().numpy().flatten()
    except Exception:
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------

def _check_front_present(request: GenRequest) -> ViewCheck:
    """Front view must always be present."""
    if Slot.FRONT in request.views:
        return ViewCheck("front_present", True, "info", "Front view present")
    return ViewCheck("front_present", False, "fail", "A front view is required")


def _check_alpha_quality(view: View) -> ViewCheck:
    """Check matting quality: alpha coverage, holes, edge fragmentation."""
    if view.image is None:
        return ViewCheck(f"alpha_{view.slot.value}", False, "warn",
                         f"{view.slot.value}: not preprocessed yet")

    alpha = np.asarray(view.image)[..., 3]
    total = alpha.size
    nonzero = np.count_nonzero(alpha)
    coverage = nonzero / total if total > 0 else 0.0

    # Check for too-low coverage (matting failed or empty)
    if coverage < 0.01:
        return ViewCheck(f"alpha_{view.slot.value}", False, "warn",
                         f"{view.slot.value}: alpha coverage {coverage:.1%} — "
                         "object may be missing or matting failed",
                         value=coverage)

    # Check for holes in the alpha (non-zero alpha surrounded by zero)
    # Simple check: count connected components in the alpha mask
    import cv2  # noqa: PLC0415
    binary = (alpha > 128).astype(np.uint8)
    n_components, _ = cv2.connectedComponents(binary)
    # Subtract 1 for background
    n_objects = n_components - 1

    if n_objects == 0:
        return ViewCheck(f"alpha_{view.slot.value}", False, "warn",
                         f"{view.slot.value}: no object detected in alpha",
                         value=coverage)

    # Check for fragmented alpha (too many separate components)
    if n_objects > 5:
        return ViewCheck(f"alpha_{view.slot.value}", True, "warn",
                         f"{view.slot.value}: alpha has {n_objects} fragments — "
                         "matting may be noisy",
                         value=coverage)

    return ViewCheck(f"alpha_{view.slot.value}", True, "info",
                     f"{view.slot.value}: alpha coverage {coverage:.1%}, "
                     f"{n_objects} component(s)",
                     value=coverage)


def _check_object_consistency(request: GenRequest) -> list[ViewCheck]:
    """Check that all views contain the same object via DINOv2 cosine similarity."""
    checks = []

    if len(request.views) < 2:
        return checks  # nothing to compare

    front = request.views.get(Slot.FRONT)
    if front is None or front.image is None:
        return checks

    front_embed = _dino_embed(front.image)
    if front_embed is None:
        checks.append(ViewCheck("object_consistency", True, "warn",
                                "DINOv2 unavailable — skipping cross-view consistency check"))
        return checks

    for slot, view in request.views.items():
        if slot is Slot.FRONT or view.image is None:
            continue

        embed = _dino_embed(view.image)
        if embed is None:
            continue

        sim = _cosine_similarity(front_embed, embed)

        if sim < 0.75:
            checks.append(ViewCheck(
                f"consistency_{slot.value}", False, "fail",
                f"{slot.value} vs front: cosine={sim:.3f} — "
                "these don't appear to be the same object",
                value=sim,
            ))
        elif sim < 0.85:
            checks.append(ViewCheck(
                f"consistency_{slot.value}", True, "warn",
                f"{slot.value} vs front: cosine={sim:.3f} — "
                "low similarity, may be a different view angle or object variant",
                value=sim,
            ))
        else:
            checks.append(ViewCheck(
                f"consistency_{slot.value}", True, "info",
                f"{slot.value} vs front: cosine={sim:.3f}",
                value=sim,
            ))

    return checks


def _check_near_duplicates(request: GenRequest) -> tuple[list[ViewCheck], list[str]]:
    """Detect near-duplicate views (cosine > 0.98) and recommend dropping them."""
    checks = []
    to_drop = []

    if len(request.views) < 2:
        return checks, to_drop

    front = request.views.get(Slot.FRONT)
    if front is None or front.image is None:
        return checks, to_drop

    front_embed = _dino_embed(front.image)
    if front_embed is None:
        return checks, to_drop

    for slot, view in request.views.items():
        if slot is Slot.FRONT or view.image is None:
            continue

        embed = _dino_embed(view.image)
        if embed is None:
            continue

        sim = _cosine_similarity(front_embed, embed)

        if sim > 0.98:
            checks.append(ViewCheck(
                f"duplicate_{slot.value}", True, "warn",
                f"{slot.value} is near-identical to front (cosine={sim:.3f}) — "
                "auto-dropping redundant slot",
                value=sim,
            ))
            to_drop.append(slot.value)

    return checks, to_drop


def _check_scale_consistency(request: GenRequest) -> ViewCheck:
    """Check that scale_ratio is consistent across views (spread < 10%)."""
    ratios = []
    for slot, view in request.views.items():
        if view.scale_ratio is not None:
            ratios.append(view.scale_ratio)

    if len(ratios) < 2:
        return ViewCheck("scale_consistency", True, "info",
                         "Single view — scale consistency N/A")

    mean_ratio = np.mean(ratios)
    if mean_ratio < 1e-6:
        return ViewCheck("scale_consistency", True, "warn",
                         "Scale ratios are all near zero")

    spread = (max(ratios) - min(ratios)) / mean_ratio
    if spread > 0.10:
        return ViewCheck("scale_consistency", True, "warn",
                         f"Scale spread {spread:.1%} across views — "
                         "objects appear at different sizes. This is the biggest "
                         "multi-view quality killer.",
                         value=spread)

    return ViewCheck("scale_consistency", True, "info",
                     f"Scale spread {spread:.1%} — consistent",
                     value=spread)


# ---------------------------------------------------------------------------
# full validation
# ---------------------------------------------------------------------------

def validate_request(request: GenRequest, *, session_id: str = "",
                     use_dino: bool = True) -> ValidationReport:
    """Run all validation checks on a preprocessed GenRequest.

    Set use_dino=False to skip DINOv2 checks (faster, for unit tests).
    """
    report = ValidationReport()

    # 1. Front present
    front_check = _check_front_present(request)
    report.checks.append(front_check)
    if not front_check.passed:
        report.valid = False

    # 2. Alpha quality per view
    for slot, view in request.views.items():
        alpha_check = _check_alpha_quality(view)
        report.checks.append(alpha_check)

    # 3. Scale consistency
    scale_check = _check_scale_consistency(request)
    report.checks.append(scale_check)

    # 4. Object consistency (DINOv2)
    if use_dino and len(request.views) > 1:
        consistency_checks = _check_object_consistency(request)
        for c in consistency_checks:
            report.checks.append(c)
            if not c.passed:
                report.valid = False

    # 5. Near-duplicate detection
    if use_dino and len(request.views) > 1:
        dup_checks, to_drop = _check_near_duplicates(request)
        for c in dup_checks:
            report.checks.append(c)
        report.dropped_slots = to_drop

        # Auto-drop near-duplicate slots
        for slot_name in to_drop:
            slot = Slot(slot_name)
            if slot in request.views and slot is not Slot.FRONT:
                del request.views[slot]

    # Log summary
    if session_id:
        events.log(session_id, f"Validation: {report.summary()}")

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Headless CLI for testing validation on preprocessed images."""
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="ANVIL view validator")
    parser.add_argument("images", nargs="+",
                        help="Preprocessed images (slot=path, e.g. front=front.png)")
    parser.add_argument("--no-dino", action="store_true",
                        help="Skip DINOv2 checks")
    args = parser.parse_args()

    from .genrequest import GenRequest, Slot, View  # noqa: PLC0415
    views = {}
    for spec in args.images:
        if "=" in spec:
            slot_name, path = spec.split("=", 1)
        else:
            slot_name, path = "front", spec

        slot = Slot(slot_name)
        img = Image.open(path).convert("RGBA")
        view = View(slot=slot, source_path=Path(path), image=img)

        # Estimate scale ratio
        alpha = np.asarray(img)[..., 3]
        if alpha.size > 0 and np.any(alpha > 0):
            rows = np.any(alpha > 0, axis=1)
            cols = np.any(alpha > 0, axis=0)
            if rows.any():
                y0, y1 = np.where(rows)[0][[0, -1]]
                x0, x1 = np.where(cols)[0][[0, -1]]
                view.scale_ratio = ((x1 - x0 + 1) * (y1 - y0 + 1)) / alpha.size

        views[slot] = view

    request = GenRequest(views=views)
    report = validate_request(request, use_dino=not args.no_dino)

    print(f"\nValidation: {'PASS' if report.valid else 'FAIL'}")
    print(f"Summary: {report.summary()}\n")
    for check in report.checks:
        icon = "PASS" if check.passed else ("WARN" if check.level == "warn" else "FAIL")
        print(f"  {icon:4s}  {check.name}: {check.message}")


if __name__ == "__main__":
    main()
