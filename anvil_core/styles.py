"""Style preset table (§5 of the implementation plan).

14 fixed-look presets + Normal (deliberate no-op) + Custom (derived per-run from
a reference image or text via caption_style_reference).

Each preset carries both the *prompt-side* modifier (what gets merged into the
generation prompt) and the *mesh-side* postprocess chain (what gets run on the
raw generated mesh). Keeping both on one record is what makes a preset a single
switch rather than two settings the user has to keep in sync.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

Category = Literal["object", "human", "creature", "any"]


@dataclass(frozen=True)
class StylePreset:
    id: str
    label: str
    category: Category
    prompt_modifier: str
    target_tris: int
    texture_size: int
    pbr: str
    # Ordered postprocess chain — names resolve to functions in postprocess.py
    postprocess: tuple[str, ...]
    remesh: bool = True
    notes: str = ""

    def as_dict(self) -> dict:
        d = asdict(self)
        d["postprocess"] = list(self.postprocess)
        return d


STYLE_PRESETS: tuple[StylePreset, ...] = (
    StylePreset(
        id="hyper_realistic",
        label="Hyper realistic",
        category="object",
        prompt_modifier=(
            "photorealistic, physically accurate materials, fine surface detail, "
            "subtle imperfections, high dynamic range lighting, 8k reference photography"
        ),
        target_tris=48000, texture_size=4096, pbr="full",
        postprocess=("remesh",), remesh=True,
    ),
    StylePreset(
        id="realistic",
        label="Realistic",
        category="object",
        prompt_modifier="realistic materials, believable proportions, natural lighting, moderate surface detail",
        target_tris=22000, texture_size=2048, pbr="full_simplified",
        postprocess=("remesh", "decimate"), remesh=True,
    ),
    StylePreset(
        id="stylized",
        label="Stylized",
        category="object",
        prompt_modifier="stylized game asset, exaggerated forms, clean readable silhouette, hand-painted feel",
        target_tris=9000, texture_size=1024, pbr="simplified",
        postprocess=("remesh", "decimate", "downsample_texture"), remesh=True,
    ),
    StylePreset(
        id="ps2",
        label="PS2 era",
        category="object",
        prompt_modifier="early-2000s console game asset, low polygon count, simple baked textures, flat shading",
        target_tris=3800, texture_size=512, pbr="off",
        postprocess=("decimate", "downsample_texture", "flatten_normals"), remesh=False,
        notes="No remesh — the chunky triangulation is the look.",
    ),
    StylePreset(
        id="ps1",
        label="PS1 era",
        category="object",
        prompt_modifier="mid-90s console game asset, extremely low poly, blocky, 128px textures, no smoothing",
        target_tris=420, texture_size=128, pbr="off",
        postprocess=("decimate", "downsample_texture", "flatten_normals"), remesh=False,
    ),
    StylePreset(
        id="anime_cel",
        label="Anime cel-shaded",
        category="any",
        prompt_modifier="anime cel-shaded, flat color regions, crisp ink outlines, minimal gradient, toon shading",
        target_tris=8600, texture_size=1024, pbr="off",
        postprocess=("remesh", "decimate", "flatten_normals"), remesh=True,
    ),
    StylePreset(
        id="hard_surface_scifi",
        label="Hard-surface sci-fi",
        category="object",
        prompt_modifier=(
            "hard-surface sci-fi, panel lines, bevelled mechanical edges, greebles, "
            "brushed metal and matte composite, industrial design language"
        ),
        target_tris=26000, texture_size=2048, pbr="full_normal_ao",
        postprocess=("remesh",), remesh=True,
    ),
    StylePreset(
        id="painterly_ghibli",
        label="Painterly (Ghibli)",
        category="any",
        prompt_modifier="soft painterly render, watercolour texture, warm muted palette, gentle organic forms, baked lighting",
        target_tris=10200, texture_size=2048, pbr="off_baked",
        postprocess=("remesh", "decimate"), remesh=True,
    ),
    StylePreset(
        id="human_stylized",
        label="Human — stylized",
        category="human",
        prompt_modifier="stylized humanoid character, appealing proportions, clean topology-friendly forms, T-pose, full body",
        target_tris=12400, texture_size=2048, pbr="simplified",
        postprocess=("remesh", "decimate"), remesh=True,
    ),
    StylePreset(
        id="human_realistic",
        label="Human — realistic",
        category="human",
        prompt_modifier="realistic human character, anatomically correct proportions, T-pose, full body, neutral expression",
        target_tris=34000, texture_size=4096, pbr="full",
        postprocess=("remesh",), remesh=True,
    ),
    StylePreset(
        id="creature_stylized",
        label="Creature — stylized",
        category="creature",
        prompt_modifier="stylized creature, exaggerated anatomy, clear silhouette, neutral standing pose, full body",
        target_tris=13100, texture_size=2048, pbr="simplified",
        postprocess=("remesh", "decimate"), remesh=True,
    ),
    StylePreset(
        id="creature_realistic",
        label="Creature — realistic",
        category="creature",
        prompt_modifier="realistic creature, believable anatomy and musculature, neutral standing pose, full body",
        target_tris=31500, texture_size=4096, pbr="full",
        postprocess=("remesh",), remesh=True,
    ),
    StylePreset(
        id="voxel_gmod",
        label="Voxel (Gmod prop)",
        category="object",
        prompt_modifier="chunky voxel prop, coarse cubic volumes, flat untextured colour blocks",
        target_tris=1900, texture_size=512, pbr="off",
        postprocess=("voxelize", "decimate", "flatten_normals"), remesh=False,
    ),
    StylePreset(
        id="minecraft_blocky",
        label="Minecraft blocky",
        category="object",
        prompt_modifier="grid-locked cubic blocks, 16px pixel-art texel density, hard axis-aligned faces, no curves",
        target_tris=620, texture_size=32, pbr="off",
        postprocess=("voxelize", "decimate", "downsample_texture", "flatten_normals"), remesh=False,
    ),
    StylePreset(
        id="normal",
        label="Normal (no style)",
        category="any",
        prompt_modifier="",
        target_tris=15000, texture_size=1024, pbr="default",
        postprocess=(), remesh=False,
        notes="Deliberate no-op — passes the backend's raw output straight through.",
    ),
    StylePreset(
        id="custom",
        label="Custom…",
        category="any",
        prompt_modifier="",  # filled at runtime by caption_style_reference()
        target_tris=14000, texture_size=2048, pbr="derived",
        postprocess=("remesh", "decimate"), remesh=True,
        notes="prompt_modifier is supplied per-generation from a reference image or text.",
    ),
)

_BY_ID = {p.id: p for p in STYLE_PRESETS}

# Presets that are meaningless / actively misleading under procedural mode (§2.6.1)
PROCEDURAL_UNSUITED = {
    "realistic", "hyper_realistic", "anime_cel", "painterly_ghibli",
    "human_realistic", "human_stylized", "creature_realistic", "creature_stylized",
}


def get_style(style_id: str) -> StylePreset:
    if style_id not in _BY_ID:
        raise KeyError(f"Unknown style preset: {style_id!r}")
    return _BY_ID[style_id]


def all_styles() -> list[dict]:
    return [p.as_dict() for p in STYLE_PRESETS]
