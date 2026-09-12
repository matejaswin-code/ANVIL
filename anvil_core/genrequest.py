"""Shape-stage request types.

One request type, discriminated by mode, because the two generation modes are
mutually exclusive rather than additive: `2mv` conditions on up to four views,
Omni conditions on one view plus exactly one control signal. Encoding that as a
single validated object keeps the exclusivity in one place instead of leaking a
mode check into every downstream branch.

A single uploaded image becomes `{Slot.FRONT: View(...)}` under MULTIVIEW and
takes the standard single-image code path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal


class ModeConflict(ValueError):
    """Raised when a request mixes the two modes. Message is user-facing."""


class Mode(str, Enum):
    MULTIVIEW = "multiview"
    CONTROLLED = "controlled"


class Slot(str, Enum):
    """The four view slots.

    A horizontal ring at fixed elevation — deliberately no TOP/BOTTOM. The
    checkpoint has nowhere to route them, so offering them in the UI would
    promise something the backend cannot honour. Top geometry is inferred from
    the four silhouettes and is the weakest region of the result.
    """

    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"


class ControlKind(str, Enum):
    BBOX = "bbox"
    VOXEL = "voxel"
    POINTS = "pointcloud"
    SKELETON = "skeleton"


@dataclass
class View:
    """One conditioning image bound to a slot.

    `image` is the matted RGBA the adapter is handed. It stays None until the
    preprocess step fills it in (plan §3), so a request can be assembled and
    validated before any pixels are touched — Phase 1 constructs these from
    paths alone.
    """

    slot: Slot
    source_path: Path
    image: Any | None = None          # PIL.Image.Image once matted
    scale_ratio: float | None = None  # object bbox / canvas, set by preprocess

    def __post_init__(self) -> None:
        self.source_path = Path(self.source_path)


@dataclass
class ControlSignal:
    kind: ControlKind
    payload_path: Path                # .npy / .ply / .json, unit-cube normalized
    source: Literal["blender", "upload", "manual"] = "upload"

    def __post_init__(self) -> None:
        self.payload_path = Path(self.payload_path)


@dataclass
class GenRequest:
    mode: Mode = Mode.MULTIVIEW
    views: dict[Slot, View] = field(default_factory=dict)
    control: ControlSignal | None = None

    # ---- construction -------------------------------------------------

    @classmethod
    def from_single_image(cls, image_path: str | Path) -> GenRequest:
        """The default path: one image lands in FRONT under MULTIVIEW.

        `2mv` has no separate single-image code path — a one-image run is simply
        a dict with only `front` populated — so this is the same call the
        four-view case makes, not a special case beside it.
        """
        path = Path(image_path)
        return cls(
            mode=Mode.MULTIVIEW,
            views={Slot.FRONT: View(slot=Slot.FRONT, source_path=path)},
        )

    # ---- invariants ---------------------------------------------------

    def validate(self) -> None:
        """Reject impossible combinations before a checkpoint loads.

        Raises ModeConflict with a message written for the user, not a stack
        trace. This is the only guard that has to hold; everything downstream
        may assume the request is coherent.
        """
        if self.mode is Mode.CONTROLLED:
            if len(self.views) > 1:
                raise ModeConflict(
                    "Controlled mode takes a single image plus one control signal. "
                    f"Got {len(self.views)} views — remove the extra ones, or switch "
                    "to Multi-view."
                )
            if self.control is None:
                raise ModeConflict(
                    "Controlled mode needs a control signal (bounding box, voxel, "
                    "point cloud or skeleton)."
                )
        elif self.mode is Mode.MULTIVIEW:
            if self.control is not None:
                raise ModeConflict(
                    "Multi-view mode takes views only. Switch to Controlled to use a "
                    "control signal."
                )

        if Slot.FRONT not in self.views:
            raise ModeConflict("A front view is required.")

        for slot, view in self.views.items():
            if view.slot is not slot:
                raise ModeConflict(
                    f"View filed under {slot.value} is tagged {view.slot.value}."
                )

    # ---- adapter boundary ---------------------------------------------

    def to_pipeline_image(self) -> Any:
        """What the shape adapter is handed.

        `2mv` wants a slot-keyed dict of PIL images. Preprocess must have run —
        an un-matted request would otherwise reach OpenCV as None and fail deep
        inside the vendored code rather than here.
        """
        missing = [s.value for s, v in self.views.items() if v.image is None]
        if missing:
            raise ModeConflict(
                f"These views haven't been preprocessed yet: {', '.join(missing)}."
            )
        return {slot.value: view.image for slot, view in self.views.items()}

    # ---- persistence (session sidecar, plan §7) -----------------------

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "views": {
                slot.value: {
                    "source_path": str(v.source_path),
                    "scale_ratio": v.scale_ratio,
                }
                for slot, v in self.views.items()
            },
            "control": None if self.control is None else {
                "kind": self.control.kind.value,
                "payload_path": str(self.control.payload_path),
                "source": self.control.source,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> GenRequest:
        views = {}
        for name, v in (data.get("views") or {}).items():
            slot = Slot(name)
            views[slot] = View(
                slot=slot,
                source_path=Path(v["source_path"]),
                scale_ratio=v.get("scale_ratio"),
            )
        control = None
        if data.get("control"):
            c = data["control"]
            control = ControlSignal(
                kind=ControlKind(c["kind"]),
                payload_path=Path(c["payload_path"]),
                source=c.get("source", "upload"),
            )
        return cls(mode=Mode(data.get("mode", "multiview")), views=views, control=control)
