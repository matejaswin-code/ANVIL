"""Data schemas (§4 of the implementation plan).

Three input modes, each with a different contract:
  Mode 1 (text_to_image) — mandatory checklist; missing fields BLOCK generation.
  Mode 2 (text)          — everything optional; generate with whatever is given.
  Mode 3 (image_upload)  — no prompt text at all; the fields are generation FLAGS.

The session object is what gets checkpointed to session.json after every step.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

Mode = Literal["text_to_image", "text", "image_upload"]

# --------------------------------------------------------------------------
# §4.1 / §4.2 / §4.3 — field definitions, consumed by the UI to build the form
# --------------------------------------------------------------------------

MODE_SCHEMAS: dict[str, dict[str, Any]] = {
    "text_to_image": {
        "label": "Text → Image",
        "hint": "mandatory checklist",
        "mandatory": [
            {"key": "object_type", "label": "Object type", "placeholder": "rusty helmet"},
            {"key": "style_desc", "label": "Style descriptor", "placeholder": "battle-worn, medieval"},
            {"key": "primary_material", "label": "Primary material", "placeholder": "weathered steel"},
            {"key": "color_palette", "label": "Color palette", "placeholder": "iron gray, rust orange"},
        ],
        "optional": [
            {"key": "lighting", "label": "Lighting", "placeholder": "dramatic side light"},
            {"key": "scale_reference", "label": "Scale reference", "placeholder": "fits in one hand"},
            {"key": "wear_and_tear", "label": "Wear and tear", "placeholder": "dented, scratched"},
            {"key": "background_context", "label": "Background context", "placeholder": "dungeon floor"},
        ],
    },
    "text": {
        "label": "Direct text",
        "hint": "all fields optional",
        "mandatory": [],
        "optional": [
            {"key": "object_type", "label": "Object type", "placeholder": "rusty helmet"},
            {"key": "style_desc", "label": "Style descriptor", "placeholder": "battle-worn, medieval"},
            {"key": "primary_material", "label": "Primary material", "placeholder": "weathered steel"},
            {"key": "color_palette", "label": "Color palette", "placeholder": "iron gray, rust orange"},
            {"key": "scale_reference", "label": "Scale reference", "placeholder": "fits in one hand"},
            {"key": "symmetry", "label": "Symmetry", "placeholder": "bilateral"},
        ],
    },
    "image_upload": {
        "label": "Image upload",
        "hint": "generation flags",
        "mandatory": [],
        "optional": [],
        "flags": [
            {"key": "background_removal", "label": "Background removal", "type": "bool", "default": True},
            {"key": "orientation_lock", "label": "Orientation lock", "type": "enum",
             "options": ["free", "upright", "flat"], "default": "free"},
            {"key": "symmetry_enforcement", "label": "Symmetry enforcement", "type": "bool", "default": False},
            {"key": "detail_level", "label": "Detail level", "type": "enum",
             "options": ["low", "medium", "high-fidelity"], "default": "high-fidelity"},
            {"key": "generate_pbr", "label": "Generate PBR maps", "type": "bool", "default": True},
        ],
    },
}

DEFAULT_FLAGS = {f["key"]: f["default"] for f in MODE_SCHEMAS["image_upload"]["flags"]}


def mandatory_keys(mode: str) -> list[str]:
    return [f["key"] for f in MODE_SCHEMAS.get(mode, {}).get("mandatory", [])]


def missing_mandatory(mode: str, checklist: dict) -> list[str]:
    """Which mandatory fields are still unfilled. Empty list == ready to generate."""
    out = []
    for key in mandatory_keys(mode):
        val = (checklist or {}).get(key)
        if not val or not str(val).strip():
            out.append(key)
    return out


# --------------------------------------------------------------------------
# §7 — the 10 pipeline steps
# --------------------------------------------------------------------------

PIPELINE_STEPS = [
    "mode_select",
    "style_resolve",
    "checklist_fill",
    "build_prompt",
    "image_generate",
    "determine_3d_input",
    "model_generate",
    "postprocess",
    "import_to_blender",
    "edit_loop",
]
STEP_INDEX = {name: i + 1 for i, name in enumerate(PIPELINE_STEPS)}


# --------------------------------------------------------------------------
# §4.4 — session object
# --------------------------------------------------------------------------

@dataclass
class EditRecord:
    instruction: str
    category: str          # property | shape | new_model | rig
    undoable: bool
    undo_params: dict | None = None
    at: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    mode: str = "text_to_image"
    style_id: str = "realistic"
    procedural: bool = False

    checklist: dict = field(default_factory=dict)
    flags: dict = field(default_factory=lambda: dict(DEFAULT_FLAGS))
    custom_style_modifier: str = ""       # filled by caption_style_reference()
    uploaded_image_path: str | None = None

    # step outputs
    final_prompt: str = ""
    # Set when the user edits the prompt during concept-image review. Step 4 then
    # leaves final_prompt alone instead of rebuilding it from the checklist and
    # silently discarding the edit.
    prompt_override: bool = False
    generated_image_path: str | None = None
    three_d_input_kind: str = ""          # "image" | "text"
    raw_mesh_path: str | None = None
    final_mesh_path: str | None = None    # or "in_scene:<object_name>" for procedural
    blend_file_path: str | None = None
    scene_object_uuid: str | None = None

    # progress
    current_step: int = 0                 # highest completed step (1-10)
    completed_steps: list[str] = field(default_factory=list)
    status: str = "new"                   # new|running|awaiting_review|done|failed|stopped
    error: str | None = None

    # edit phase
    edit_history: list[dict] = field(default_factory=list)
    exported_paths: dict = field(default_factory=dict)
    regenerate_count: int = 0

    # batch linkage
    batch_id: str | None = None
    label: str = ""

    def mark_step(self, step_name: str) -> None:
        if step_name not in self.completed_steps:
            self.completed_steps.append(step_name)
        self.current_step = max(self.current_step, STEP_INDEX[step_name])
        self.updated_at = time.time()

    def has_completed(self, step_name: str) -> bool:
        return step_name in self.completed_steps

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Session":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def display_label(self) -> str:
        if self.label:
            return self.label
        if self.mode == "image_upload" and self.uploaded_image_path:
            from pathlib import Path
            return Path(self.uploaded_image_path).name
        return self.checklist.get("object_type") or self.checklist.get("style_desc") or "(untitled)"
