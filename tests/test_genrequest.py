"""Mode-exclusivity and request-shape tests (plan §14).

The point of these is that a bad combination is rejected at validate(), not deep
inside a vendored adapter after a multi-GB checkpoint has already loaded.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anvil_core.genrequest import (  # noqa: E402
    ControlKind, ControlSignal, GenRequest, Mode, ModeConflict, Slot, View,
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


def _view(slot):
    return View(slot=slot, source_path=Path(f"{slot.value}.png"))


def _rejects(req, fragment):
    try:
        req.validate()
    except ModeConflict as exc:
        assert fragment.lower() in str(exc).lower(), f"message was: {exc}"
        return
    raise AssertionError("expected ModeConflict, got none")


# --- the headline guard: views + control together ---------------------------
def t_views_plus_control_rejected():
    req = GenRequest(
        mode=Mode.MULTIVIEW,
        views={Slot.FRONT: _view(Slot.FRONT)},
        control=ControlSignal(kind=ControlKind.BBOX, payload_path=Path("b.json")),
    )
    _rejects(req, "views only")


def t_controlled_with_many_views_rejected():
    req = GenRequest(
        mode=Mode.CONTROLLED,
        views={s: _view(s) for s in (Slot.FRONT, Slot.LEFT)},
        control=ControlSignal(kind=ControlKind.BBOX, payload_path=Path("b.json")),
    )
    _rejects(req, "single image")


def t_controlled_without_control_rejected():
    req = GenRequest(mode=Mode.CONTROLLED, views={Slot.FRONT: _view(Slot.FRONT)})
    _rejects(req, "control signal")


def t_front_required():
    req = GenRequest(mode=Mode.MULTIVIEW, views={Slot.BACK: _view(Slot.BACK)})
    _rejects(req, "front view is required")


def t_slot_key_must_match_view():
    req = GenRequest(mode=Mode.MULTIVIEW, views={Slot.FRONT: _view(Slot.BACK)})
    _rejects(req, "tagged")


# --- the paths that must keep working ---------------------------------------
def t_single_image_is_multiview_front():
    req = GenRequest.from_single_image("concept.png")
    req.validate()
    assert req.mode is Mode.MULTIVIEW
    assert list(req.views) == [Slot.FRONT]
    assert req.control is None


def t_four_views_valid():
    req = GenRequest(mode=Mode.MULTIVIEW, views={s: _view(s) for s in Slot})
    req.validate()
    assert len(req.views) == 4


def t_controlled_valid():
    req = GenRequest(
        mode=Mode.CONTROLLED,
        views={Slot.FRONT: _view(Slot.FRONT)},
        control=ControlSignal(kind=ControlKind.BBOX, payload_path=Path("b.json")),
    )
    req.validate()


def t_no_top_or_bottom_slot():
    # The ring is horizontal; offering TOP/BOTTOM would promise routing that
    # does not exist in the checkpoint.
    assert {s.value for s in Slot} == {"front", "back", "left", "right"}


def t_pipeline_image_refuses_unpreprocessed():
    req = GenRequest.from_single_image("concept.png")
    try:
        req.to_pipeline_image()
    except ModeConflict as exc:
        assert "preprocessed" in str(exc)
        return
    raise AssertionError("expected refusal for un-matted views")


def t_roundtrip_preserves_mode_and_control():
    req = GenRequest(
        mode=Mode.CONTROLLED,
        views={Slot.FRONT: _view(Slot.FRONT)},
        control=ControlSignal(kind=ControlKind.VOXEL, payload_path=Path("v.npy"), source="blender"),
    )
    back = GenRequest.from_dict(req.to_dict())
    assert back.mode is Mode.CONTROLLED
    assert back.control.kind is ControlKind.VOXEL
    assert back.control.source == "blender"


if __name__ == "__main__":
    print("GenRequest — mode exclusivity and shape")
    for name, fn in sorted(globals().items()):
        if name.startswith("t_"):
            check(name[2:], fn)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        sys.exit(1)
    print("all passed")
