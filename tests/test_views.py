"""Tests for view_preprocess and view_validate modules.

Exercises the preprocessing pipeline and validation gate with synthetic
images — no GPU, no downloaded models, fully self-contained.
"""
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw
import numpy as np

from anvil_core.genrequest import GenRequest, Slot, View, Mode
from anvil_core.view_preprocess import (
    _exif_normalize, _crop_pad_center, _scale_normalize,
    _resize_to_model_input, _remove_background,
    preprocess_view, preprocess_request,
)
from anvil_core.view_validate import (
    ValidationReport, validate_request,
    _check_front_present, _check_alpha_quality, _check_scale_consistency,
)

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as exc:
        FAILURES.append(name)
        print(f"  FAIL  {name}: {exc}")
    except Exception as exc:
        FAILURES.append(name)
        print(f"  ERROR {name}: {type(exc).__name__}: {exc}")


def _make_test_image(size=256, has_alpha=False, subject_fraction=0.6):
    """Create a synthetic test image with a colored circle on dark background."""
    if has_alpha:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        margin = int(size * (1 - subject_fraction) / 2)
        draw.ellipse([margin, margin, size - margin, size - margin],
                     fill=(200, 100, 50, 255))
    else:
        img = Image.new("RGB", (size, size), (20, 20, 25))
        draw = ImageDraw.Draw(img)
        margin = int(size * (1 - subject_fraction) / 2)
        draw.ellipse([margin, margin, size - margin, size - margin],
                     fill=(200, 100, 50))
    return img


def _save_temp(img, suffix=".png"):
    """Save image to a temp file and return the path."""
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    img.save(f.name)
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# Preprocessing tests
# ---------------------------------------------------------------------------

def t_exif_normalize_returns_pil():
    img = _make_test_image()
    result = _exif_normalize(img)
    assert isinstance(result, Image.Image)
    assert result.size == img.size


def t_crop_pad_center_with_alpha():
    img = _make_test_image(256, has_alpha=True, subject_fraction=0.5)
    result, bbox = _crop_pad_center(img)
    assert result.mode == "RGBA"
    # Result should be square
    assert result.width == result.height
    # Bbox should be within original image
    x0, y0, x1, y1 = bbox
    assert x0 >= 0 and y0 >= 0
    assert x1 <= 256 and y1 <= 256


def t_crop_pad_center_empty_alpha():
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    result, bbox = _crop_pad_center(img)
    # Should return original when alpha is empty
    assert result.size == img.size


def t_scale_normalize_fills_canvas():
    img = _make_test_image(512, has_alpha=True, subject_fraction=0.3)
    result, ratio = _scale_normalize(img, target_fill=0.85)
    assert isinstance(result, Image.Image)
    assert result.mode == "RGBA"
    assert ratio > 0


def t_resize_to_model_input():
    img = _make_test_image(256, has_alpha=True)
    result = _resize_to_model_input(img, size=512)
    assert result.size == (512, 512)


def t_resize_noop_when_correct():
    img = _make_test_image(512, has_alpha=True)
    result = _resize_to_model_input(img, size=512)
    assert result is img  # should be same object, no resampling


def t_remove_background_with_existing_alpha():
    img = _make_test_image(128, has_alpha=True, subject_fraction=0.6)
    result = _remove_background(img)
    assert result.mode == "RGBA"
    # Should preserve existing alpha
    alpha = np.asarray(result)[..., 3]
    assert alpha.min() < 250  # has transparency


def t_preprocess_view_full_pipeline():
    img = _make_test_image(256, has_alpha=True, subject_fraction=0.5)
    path = _save_temp(img)
    try:
        view = View(slot=Slot.FRONT, source_path=path)
        result = preprocess_view(view)
        assert view.image is not None
        assert view.image.mode == "RGBA"
        assert view.image.size == (512, 512)
        assert view.scale_ratio is not None
        assert result.alpha_coverage > 0
    finally:
        path.unlink(missing_ok=True)


def t_preprocess_request_multi_view():
    imgs = {}
    paths = {}
    for slot in [Slot.FRONT, Slot.LEFT]:
        img = _make_test_image(200, has_alpha=True, subject_fraction=0.5)
        p = _save_temp(img)
        paths[slot] = p
        imgs[slot] = img

    try:
        views = {s: View(slot=s, source_path=p) for s, p in paths.items()}
        request = GenRequest(mode=Mode.MULTIVIEW, views=views)
        results = preprocess_request(request)
        assert len(results) == 2
        for r in results:
            assert r.view.image is not None
            assert r.view.image.size == (512, 512)
    finally:
        for p in paths.values():
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def t_validate_front_present():
    views = {Slot.FRONT: View(slot=Slot.FRONT, source_path=Path("f.png"),
                               image=_make_test_image(64, has_alpha=True))}
    request = GenRequest(mode=Mode.MULTIVIEW, views=views)
    check_result = _check_front_present(request)
    assert check_result.passed


def t_validate_front_missing():
    views = {Slot.BACK: View(slot=Slot.BACK, source_path=Path("b.png"),
                              image=_make_test_image(64, has_alpha=True))}
    request = GenRequest(mode=Mode.MULTIVIEW, views=views)
    check_result = _check_front_present(request)
    assert not check_result.passed
    assert check_result.level == "fail"


def t_validate_alpha_quality_good():
    img = _make_test_image(128, has_alpha=True, subject_fraction=0.6)
    view = View(slot=Slot.FRONT, source_path=Path("f.png"), image=img)
    check_result = _check_alpha_quality(view)
    assert check_result.passed


def t_validate_alpha_quality_empty():
    img = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    view = View(slot=Slot.FRONT, source_path=Path("f.png"), image=img)
    check_result = _check_alpha_quality(view)
    # Should warn about low alpha coverage
    assert check_result.level == "warn"


def t_validate_scale_consistency_single():
    views = {Slot.FRONT: View(slot=Slot.FRONT, source_path=Path("f.png"),
                               scale_ratio=0.5)}
    request = GenRequest(mode=Mode.MULTIVIEW, views=views)
    check_result = _check_scale_consistency(request)
    assert check_result.passed


def t_validate_scale_consistency_good():
    views = {
        Slot.FRONT: View(slot=Slot.FRONT, source_path=Path("f.png"), scale_ratio=0.50),
        Slot.LEFT: View(slot=Slot.LEFT, source_path=Path("l.png"), scale_ratio=0.52),
    }
    request = GenRequest(mode=Mode.MULTIVIEW, views=views)
    check_result = _check_scale_consistency(request)
    assert check_result.passed
    assert check_result.level == "info"


def t_validate_scale_consistency_bad():
    views = {
        Slot.FRONT: View(slot=Slot.FRONT, source_path=Path("f.png"), scale_ratio=0.50),
        Slot.LEFT: View(slot=Slot.LEFT, source_path=Path("l.png"), scale_ratio=0.20),
    }
    request = GenRequest(mode=Mode.MULTIVIEW, views=views)
    check_result = _check_scale_consistency(request)
    assert check_result.level == "warn"


def t_validate_full_request_single_view():
    img = _make_test_image(128, has_alpha=True, subject_fraction=0.6)
    views = {Slot.FRONT: View(slot=Slot.FRONT, source_path=Path("f.png"),
                               image=img, scale_ratio=0.5)}
    request = GenRequest(mode=Mode.MULTIVIEW, views=views)
    report = validate_request(request, use_dino=False)
    assert report.valid
    assert len(report.failures) == 0


def t_validate_full_request_no_front():
    img = _make_test_image(128, has_alpha=True, subject_fraction=0.6)
    views = {Slot.BACK: View(slot=Slot.BACK, source_path=Path("b.png"),
                              image=img, scale_ratio=0.5)}
    request = GenRequest(mode=Mode.MULTIVIEW, views=views)
    report = validate_request(request, use_dino=False)
    assert not report.valid
    assert len(report.failures) > 0


def t_validate_report_summary():
    report = ValidationReport()
    assert "all checks passed" in report.summary()


if __name__ == "__main__":
    print("view_preprocess + view_validate tests")
    for name, fn in sorted(globals().items()):
        if name.startswith("t_"):
            check(name[2:], fn)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all passed")
