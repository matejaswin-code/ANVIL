# ANVIL 1.1 — Implementation Plan

Scope of the 1.1 release:

- Multi-view image input (Hunyuan3D-2mv)
- Controlled generation from geometric priors (Hunyuan3D-Omni)
- Selectable image-generation backends
- LM Studio integration: prompt rewrite, image quality gate, slot assist
- Image quality techniques: best-of-N, delighting, supersampling, framing LoRA
- Material controls before texture generation
- Built-in A/B comparison harness

Backend family: Hunyuan3D. Local, personal use — ANVIL is not being distributed,
so licensing terms are not a constraint here.

Two generation modes, backed by **two different checkpoints**. They cannot be
combined in a single run.

| Mode | Checkpoint | Base | Input | VRAM |
|---|---|---|---|---|
| `MULTIVIEW` | `hunyuan3d-dit-v2-mv` / `-turbo` | 2.0 | 1–4 directional images | ~6 GB (shape) |
| `CONTROLLED` | `Hunyuan3D-Omni` | 2.1 | 1 image + 1 geometric control signal | ~10 GB |

---

## 0. Three corrections before you build

**1. The gain is geometry, not texture.**
Hunyuan3D-2 splits generation into shape (DiT) and texture (Paint). Both `2mv`
and Omni condition the *shape* stage. Back-side texture still comes from the
paint stage working off the front view. Set expectations accordingly.

**2. Pin the checkpoint version.**
Hunyuan3D-2.1 contains multi-view code but its default checkpoint has no `mv`
weights, so MV mode is off and forcing a view dict through it crashes in
preprocessing. There is no released 2.1-mv. `MULTIVIEW` must stay on the 2.0
line; Omni is separately built on 2.1. Assert the expected mode flag at load and
fail loudly.

**3. Modes are exclusive, not additive.**
You cannot feed four views *and* a point cloud. Omni also trains with one control
modality per example, so you don't stack two control signals either. The UI must
make this a choice, not a set of checkboxes.

Checkpoint default for `MULTIVIEW`: **`-turbo`** (step-distilled, lower VRAM),
with full quality as an option.

---

## 1. Backend contracts

### MULTIVIEW — `hunyuan3d-dit-v2-mv`

Slot-keyed dict, not a list:

```python
pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mv",
    subfolder="hunyuan3d-dit-v2-mv",
    use_safetensors=True,
    device="cuda",
)
mesh = pipeline(
    image={"front": ..., "left": ..., "back": ..., "right": ...},
    num_inference_steps=30,
    octree_resolution=380,
    num_chunks=20000,
)
```

Only `front` is mandatory; the rest degrade gracefully as you omit them. **There
is no separate single-image code path** — a one-image run is a dict with only
`front` populated.

**No top or bottom slot.** The four slots are a horizontal ring at fixed
elevation. Do not add `TOP`/`BOTTOM` to the enum — there is nowhere to route
them. Top geometry is inferred from the four silhouettes and will be the weakest
region of the mesh; plan to fix it in Blender if it matters.

### CONTROLLED — Hunyuan3D-Omni

Image plus exactly one control signal. Omni represents every modality internally
as a point cloud through a unified control encoder, which is why the four look
so different at the UI layer but converge in the backend.

| Signal | What it constrains | Where it comes from in ANVIL |
|---|---|---|
| Bounding box | Aspect ratio and scale, via eight canonical-space vertices | Numeric L/W/H field, or a Blender empty/cube |
| Voxel | Coarse occupancy — the blockout | Remesh a rough Blender model |
| Point cloud | Spatial structure; accepts partial and noisy input | Surface-sample a mesh, a scan, or a depth projection |
| Skeleton | Pose, for humanoids | Blender armature export |

Prioritize **bounding box** — it's the cheapest to build (a few numbers, no new
geometry pipeline) and it solves the most common real complaint, that generated
assets come out at arbitrary proportions.

Skeleton pairs naturally with UniRig already being in the stack: pose in, rig
after.

---

## 1a. Image generator registry

The image stage becomes a selectable backend the same way the shape stage is.
Same pattern, separate axis — you pick an image model *and* a shape mode.

```python
class ImageBackend(Protocol):
    name: str
    vram_estimate_gb: float
    native_resolution: tuple[int, int]
    default_steps: int
    default_guidance: float
    supports_negative: bool

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def generate(self, prompt: str, negative: str, seed: int,
                 count: int = 1, **kw) -> list[Image.Image]: ...
```

Candidate entries:

| Backend | Size | License | Note |
|---|---|---|---|
| FLUX.1 [dev] | 12B | BFL, paid for commercial | Strong prompt adherence; current default |
| FLUX.2 [dev] | large | BFL, paid for commercial | Leads open-weight prompt fidelity |
| FLUX.2 [klein] | 4B | Apache 2.0 | Low-latency, ~13 GB |
| Qwen-Image-2512 | 20B | Apache 2.0 | Latest confirmed open weights; GGUF quants exist |
| Qwen-Image Lightning | distilled | Apache 2.0 | 4–8 steps, large speedup, minor quality cost |
| Z-Image Turbo | 6B | — | ~1s generation; VRAM-tight fallback |
| SDXL 1.0 | 3.5B | — | Deepest LoRA ecosystem — relevant for §7's framing LoRA |

Licensing is not a blocker (ANVIL isn't distributed), but record it per entry
anyway so the decision is visible if that ever changes.

### Adapter quirks — the reason this needs a real abstraction

Models are not drop-in interchangeable:

- **Qwen-Image uses `true_cfg_scale`, not the usual guidance parameter,** and its
  reference usage appends a "positive magic" suffix per language. It also expects
  resolution from a fixed aspect-ratio bucket table rather than arbitrary
  dimensions.
- **FLUX variants differ in negative-prompt support.** `supports_negative` is on
  the protocol for a reason — silently dropping your 3D-safety negative on a
  model that ignores it is a quiet quality regression.
- **Distilled variants need their own step/CFG defaults.** Running Lightning at 40
  steps wastes time; running full Qwen-Image at 4 steps produces mush.

Per-backend defaults live with the backend, not in the preset. Presets override
where they need to.

### Selection and VRAM

- Global default image backend, overridable per style preset.
- One image backend resident at a time — another entry in the single-resident
  registry alongside the shape checkpoints, matting model, and LM Studio.
- Quantization (GGUF / FP8) is a per-entry config field, not a separate backend.

---

## 2. Core types

```python
class Mode(str, Enum):
    MULTIVIEW  = "multiview"
    CONTROLLED = "controlled"

class Slot(str, Enum):
    FRONT = "front"
    BACK  = "back"
    LEFT  = "left"
    RIGHT = "right"

class ControlKind(str, Enum):
    BBOX     = "bbox"
    VOXEL    = "voxel"
    POINTS   = "pointcloud"
    SKELETON = "skeleton"

@dataclass
class View:
    image: Image.Image          # RGBA after matting
    slot: Slot
    source_path: Path
    scale_ratio: float | None = None

@dataclass
class ControlSignal:
    kind: ControlKind
    payload_path: Path          # .npy / .ply / .json, normalized to unit cube
    source: Literal["blender", "upload", "manual"]

@dataclass
class GenRequest:
    mode: Mode
    views: dict[Slot, View]                 # front always present
    control: ControlSignal | None = None    # present iff mode is CONTROLLED

    def validate(self) -> None:
        if self.mode is Mode.CONTROLLED and len(self.views) > 1:
            raise ModeConflict("Omni takes a single image plus one control signal")
        if self.mode is Mode.MULTIVIEW and self.control is not None:
            raise ModeConflict("2mv takes views only")
```

One request type, mode-discriminated. The `validate()` guard is what keeps the
exclusivity from leaking into every downstream branch.

---

## 3. Image preprocessing

Identical for every view, in both modes. Divergence between views is the main
failure mode.

1. EXIF-normalize on load — rotation metadata silently flips side views.
2. Background removal: same model, same threshold, every view.
3. Crop to object bbox, pad square, center.
4. Scale-normalize so the object fills the same canvas fraction (~0.85) in all
   views. Mismatched scale is the biggest quality killer.
5. Resize to the model's expected input resolution.

The OpenCV failure in the 2.1 issue was a type error in preprocessing — pass real
PIL images, not paths or arrays, and assert types at the adapter boundary.

---

## 4. Control-signal preprocessing (CONTROLLED only)

Every kind normalizes to the model's canonical space before encoding.

- **BBOX** — eight vertices in canonical space. Accept L/W/H, derive the rest.
  Validate no zero/negative dimension.
- **VOXEL** — binary occupancy grid. Enforce a fixed resolution; downsample
  rather than reject an oversized grid.
- **POINTS** — subsample to the model's expected count. Center and scale to the
  unit cube. Partial and noisy clouds are explicitly supported, so don't try to
  hole-fill first.
- **SKELETON** — joint positions in canonical space. Validate the joint count and
  ordering against the expected convention before submitting; a silently
  mismatched skeleton produces confidently wrong poses.

All four end up as point clouds internally, so a single normalization helper
(center → scale to unit cube → dtype/shape check) covers most of this.

---

## 5. Validation gate

Blocks bad input before a heavy checkpoint loads.

**Views (both modes):**

| Check | Method | On fail |
|---|---|---|
| Front present | slot map | Hard fail |
| Same object across slots | DINOv2/CLIP cosine vs front | Hard fail below ~0.75 |
| Scale consistency | `scale_ratio` spread post-normalization | Warn > 10% |
| Lighting consistency | masked luminance + histogram distance | Warn, offer drop |
| Matting quality | alpha coverage, holes, edge fragmentation | Warn per view |
| Near-duplicate slots | cosine > 0.98 | Auto-drop the redundant slot |

**Control signal (CONTROLLED):**

| Check | On fail |
|---|---|
| Mode/payload agreement | Hard fail |
| Payload parses, correct dtype and shape | Hard fail |
| Non-degenerate extent (not flat, not empty) | Hard fail |
| Rough silhouette agreement with the front image | Warn — the control probably isn't the same object |

Returns a `GenRequestReport` the UI renders. Not optional: with inconsistent
views, three slots produce a worse mesh than one clean front image.

---

## 6. Slot assignment

Manual only. Four labelled tiles; the user drops images in. The model is
slot-semantic, so guessing is worse than asking, and auto-pose estimation would
burn VRAM the resource manager doesn't have spare.

Optional later: a VLM *suggests* a label the user confirms (§6a). Never
auto-apply.

---

## 6a. LM Studio (prompt rewrite, image quality gate, slot assist)

LM Studio serves LLMs and VLMs, not diffusion models. It does **not** generate
images — Qwen-Image/FLUX still run through diffusers or ComfyUI. It earns its
place at three points, all OpenAI-compatible HTTP calls, so this is a client
change rather than an architecture change.

### Role 1 — prompt rewrite (text-driven runs)

Turn the user's prompt into a 3D-safe image prompt. The constraints below are
enforced programmatically rather than left to the user's phrasing:

- flat, diffuse lighting; no hard cast shadows
- no depth of field, no motion blur
- no strong speculars or reflections
- plain, high-contrast background
- full object in frame with margin
- near-orthographic framing
- single object, no ground plane, no props

Composition: `style preset prompt` + `fixed 3D-safety suffix` + `fixed negative`.
The safety layer is not user-editable. The LLM rewrites the *subject* portion
only; it must not be able to strip the suffix.

### Role 2 — image quality gate (the one worth building)

After image generation, before the shape model loads, show the image to a VLM
with a fixed checklist:

| Check | Fail action |
|---|---|
| Exactly one object present | Regenerate |
| Object fully in frame, not cropped | Regenerate |
| Background plain and separable | Regenerate |
| No hard cast shadow on the object | Regenerate |
| No blown speculars or mirror surfaces | Warn |
| In focus, no bokeh | Regenerate |

Return strict JSON (`{"pass": bool, "failures": [...]}`), not prose. Retry with
an adjusted seed, capped at N attempts — then surface the last image plus the
failure list rather than looping forever.

This buys a real quality loop without adding a second diffusion model.

### Role 3 — slot assist

Ask the VLM which side of the object a photo shows. Populates the slot strip as
a *suggestion* the user confirms. Optional; skip if Role 2 is enough value.

### Implementation notes

- Single client module wrapping the OpenAI-compatible endpoint. Config: base URL,
  model name per role, timeout, retry cap.
- **LM Studio must be treated as optional.** If the endpoint is unreachable, the
  pipeline degrades: no rewrite (use the raw prompt + safety suffix), no gate
  (pass through), no slot assist. Never block a run on it.
- Roles 1 and 3 need a text model; Role 2 needs a vision model. Don't assume one
  loaded model covers both — check and fail soft.
- Enable JIT loading and TTL auto-unload so weights release between calls.

---

## 6b. Image quality techniques

Ordered by value. All of these operate on the conditioning image, which is the
highest-leverage point in the pipeline — the shape model can only be as good as
what it's shown.

### 1. Best-of-N, not retry-on-fail

Generate 4 candidates in one batch, have the §6a gate rank them against the
checklist, take the winner. Diffusion batches efficiently, so 4 images cost far
less than 4 sequential generations. This turns the gate from a filter into a
selector — a much bigger lift for the same code.

Retry-on-fail stays as the fallback when all N fail.

### 2. Edit-model normalization pass

Instead of hoping the generator obeys the constraints, fix the image afterward:
flatten lighting, remove cast shadows, replace the background with flat neutral.

Qwen-Image-Edit-2509 is Apache 2.0 and shares the text encoder and VAE with
Qwen-Image, so it costs far less extra VRAM than an unrelated model. Delighting
the input is standard practice in 3D pipelines and improves both matting and
downstream PBR quality.

Register it as an optional post-step on the image backend, not a separate stage.

### 3. Generate high, downsample

Render at the backend's native high resolution (2K where supported), then
downsample to the shape model's input size. Supersampled edges give matting a
much cleaner alpha boundary than generating at target resolution.

This matters more than it sounds: alpha edge quality determines silhouette
accuracy, and silhouette is most of what `2mv` conditions on.

### 4. Framing LoRA

The biggest style-control lever available. Train or source one for orthographic
product-shot framing — flat lighting, centred subject, neutral backdrop. One
LoRA gets closer to 3D-safe output than any amount of prompt engineering.

SDXL has the deepest LoRA ecosystem if you want something off the shelf; the
Qwen-Image ecosystem is thinner. This is a real argument for keeping SDXL in the
§1a registry despite its age.

LoRA path and weight become per-backend config fields.

### 5. Per-preset CFG and steps

High guidance produces contrast and saturation artifacts that read as baked-in
lighting and confuse matting at the edges. Presets currently control style; they
should carry their own guidance and step counts too, layered over the
backend defaults from §1a.

### 6. Close the loop on negatives

Log which gate check fails most often, per preset and per backend. Write targeted
negative prompts from that data rather than intuition. After a few dozen runs
you'll know whether your actual problem is shadows, crops, or bokeh — and the
same log tells you when a checklist item is too strict (risk 8).

---

## 7. State machine

```
  0.  PROMPT_REWRITE      NEW  (§6a role 1, text-driven runs only, soft-fail)
  0a. IMAGE_GEN                (existing diffusion stage, best-of-N per §6b)
  0b. IMAGE_NORMALIZE     NEW  (§6b item 2, optional delighting pass)
  0c. IMAGE_GATE          NEW  (§6a role 2, ranks candidates, soft-fail)
  1.  INPUT_INGEST        -> builds GenRequest (mode + views + optional control)
  1a. VIEW_PREPROCESS     NEW  (§3, per slot)
  1b. CONTROL_PREPROCESS  NEW  (§4, CONTROLLED only, skipped otherwise)
  1c. REQUEST_VALIDATE    NEW  (§5)
  2..N (existing)         -> shape stage dispatches on mode
```

Steps 0–0c are skipped entirely when the user uploads their own images. All
soft-fail: an unreachable LM Studio degrades the run, never blocks it.

- Preprocess and validate run even for a single image — the pipeline is uniform.
- All three checkpoint: matted views persist as PNG, control payloads as-is,
  plus a `request.json` sidecar in the session dir. Resume never re-mattes.
- `REQUEST_VALIDATE` is the only step that fails on user input. Structured error,
  never a traceback.

---

## 8. Resource manager

You now have up to three shape checkpoints competing for one resident slot:
base 2.x, `dit-v2-mv(-turbo)`, and Omni. Reduce that.

- **Drop the base checkpoint.** `2mv` handles n=1 natively, so the base shape
  model is redundant the moment `MULTIVIEW` works. Benchmark parity at n=1
  first — if it holds, deleting it removes an entire evict/load axis. This is the
  single biggest simplification available.
- That leaves two: `2mv` (~6 GB shape) and Omni (~10 GB). Switching between them
  is a full evict + load, so the batch scheduler must **group queue items by
  mode** rather than running them in submission order.
- Matting is GPU-resident too. Load → process all views → evict, strictly before
  the shape checkpoint loads. Never interleave.
- **LM Studio holds its own weights outside your registry.** That's how the cloud
  LLM eviction bug happened. A VLM sitting resident during the gate will fight
  the diffusion model for the same budget. Sequence it explicitly: image gen →
  evict diffusion → load VLM → judge → unload VLM → load shape checkpoint. Use
  JIT + TTL so LM Studio releases on its own if a step fails midway.
- Register LM Studio as an entry in the resident registry even though you don't
  control its loading. Tracking it approximately beats not tracking it.
- VRAM scales with `octree_resolution` far more than with view count. Profile the
  resolution ladder per mode; that's the real budget dial.
- Omni's 10 GB is shape only. Confirm headroom for the texture stage afterward
  before advertising it as available on your card.

---

## 9. FastAPI / SSE

- `POST /session/{id}/mode` — set mode. Rejects if the run is past validation.
- `POST /session/{id}/views` — multipart, explicit slot per file.
- `DELETE|PUT /session/{id}/views/{slot}`
- `POST /session/{id}/control` — upload payload + kind. Replaces any existing
  signal (only one allowed).
- `DELETE /session/{id}/control`
- `GET /session/{id}/request` — current `GenRequest` + last report.

Events: `view.preprocess.progress` (per slot), `control.preprocess.progress`,
`request.validation.report`.

Carry forward the three bugs already fixed:
- **Path traversal** — server-generated IDs, never client filenames, writes
  confined to the session dir. Control payloads are a *new* file-write surface
  and a new parse surface — whitelist extensions and cap size.
- **Concurrency** — parallel uploads mutate shared state; keep it behind the
  session lock and reject mutation once past `REQUEST_VALIDATE`.
- **SSE pool** — per-slot progress multiplies event volume; re-check pool sizing.

---

## 10. UI

Input zone gets a mode toggle at the top: **Multi-view** | **Controlled**. It
switches the panel below rather than revealing both — the exclusivity should be
structurally obvious, not a validation error the user hits later.

**Multi-view panel:** four labelled tiles (Front marked required). Drag to
reassign. Matted preview on a transparency checkerboard so matting failures are
visible *before* generation.

**Controlled panel:** one image tile, plus a control-kind selector. Each kind
gets its own affordance — numeric L/W/H for bbox, a file drop for voxel/points/
skeleton, and a "pull from Blender" button where the addon can supply it.

Shared: validation banner with per-item chips linking to the offender. Copy
stating plainly that extra views and control signals improve **shape**, not
texture.

Single-image drop with no other input lands in Front under `MULTIVIEW` and
generates with zero extra clicks. That stays the default path.

---

## 11. Batch queue & checkpointing

- Queue item gains `mode` and a control payload reference. Bump schema version,
  migrate existing persisted queues.
- **Scheduler groups by mode** to avoid checkpoint thrash (§8).
- Session checkpoint gains `request.json` plus stored control payload.

---

## 11a. Material controls (pre-texture)

Let the user set material properties before texture generation. Worth splitting
into three tiers, because they cost very different amounts.

### Tier 1 — scalar overrides (cheap, deterministic, do these)

These are Principled BSDF inputs. They don't need to reach any model — they're
written into the material after texture generation. Changing them is instant and
requires **no regeneration**, which is the main UX win here.

| Control | Range | glTF export |
|---|---|---|
| Metallic | 0–1 | Core PBR |
| Roughness | 0–1 | Core PBR |
| Alpha / opacity | 0–1 + blend mode | Core (`alphaMode`, `alphaCutoff`) |
| Emission colour + strength | — | Core + `KHR_materials_emissive_strength` |
| IOR | 1.0–2.5 | `KHR_materials_ior` |
| Transmission | 0–1 | `KHR_materials_transmission` |
| Specular level / tint | 0–1 | `KHR_materials_specular` |
| Coat weight / roughness / IOR | 0–1 | `KHR_materials_clearcoat` |
| Sheen weight / roughness / tint | 0–1 | `KHR_materials_sheen` |
| Anisotropy + rotation | 0–1 | `KHR_materials_anisotropy` |
| Subsurface weight / radius / scale | — | No glTF equivalent — Blender only |
| Volume: density, absorption colour | — | `KHR_materials_volume` (needs transmission) |
| Normal map strength | 0–2 | Core (scale on normal texture) |

Two export targets, two fidelities:
- **Blender addon path** — set directly on the Principled BSDF. Lossless, all of
  the above including subsurface.
- **GLB path** — glTF supports a PBR subset plus extensions. Anything without a
  mapping is dropped. Warn in the UI at export rather than silently losing it,
  and verify which extensions your Blender exporter version actually writes.

### Tier 2 — appearance conditioning (must reach the texture prompt)

Some material choices change what the base colour *should* contain, so a slider
alone gives the wrong result:

- **Glass / high transmission** — the texture model must not paint refraction and
  highlights into the albedo, or you get double glass.
- **Metal** — texture models routinely bake specular highlights into base colour.
  Combined with high metallic, the highlights double up. The delighting pass
  helps; a prompt hint helps more.
- **Emissive** — glow belongs in the emission channel, not baked into albedo.

So the material selection feeds a short conditioning hint into the texture stage,
alongside the numeric values. Keep it to a fixed mapping (material type →
hint string), not free text.

### Tier 3 — out of scope

Full node graphs, procedural textures, custom shader trees. Unbounded surface,
and no useful serialization to GLB. This is what the Blender addon is for — hand
off a clean Principled setup and let Blender do the rest. Don't rebuild the
shader editor.

### Data model

```python
class MaterialType(str, Enum):
    STANDARD = "standard"
    METAL    = "metal"
    GLASS    = "glass"
    EMISSIVE = "emissive"
    FABRIC   = "fabric"
    SKIN     = "skin"

@dataclass
class MaterialSpec:
    type: MaterialType = MaterialType.STANDARD
    params: dict[str, float | tuple] = field(default_factory=dict)  # Tier 1
    # conditioning hint derived from `type`, not stored
```

Each style preset ships a default `MaterialSpec`; the user overrides per run.
Persisted in the session so re-export doesn't lose it.

### Pipeline placement

```
  ... TEXTURE_GEN
  MATERIAL_APPLY      NEW  (Tier 1 values onto the generated material)
  EXPORT
```

`MATERIAL_APPLY` and `EXPORT` must be **re-runnable without regenerating
anything**. Tweaking roughness should take a second, not a full pipeline run.
This means caching the generated maps separately from the material definition.

### Interaction warnings

- **Transparent materials break the front end.** A glass conditioning image
  confuses both matting (nothing solid to segment) and the shape model (it sees
  through the object). Generate an *opaque* proxy image, build geometry from
  that, and apply transmission afterward. Say so in the UI when GLASS is picked.
- **Alpha vs matting.** Object opacity and the RGBA alpha used for background
  removal are different things sharing a name. Keep them separate in code and in
  the UI copy.

---

---

## 12. Blender addon

This stops being a passive mesh consumer. It becomes the natural source of Omni
control signals:

- Export selection bounds → bbox.
- Remesh selection to a voxel grid → voxel payload.
- Surface-sample selection → point cloud.
- Export armature → skeleton.

That's a "send as control signal" operator writing into the session via §9's
control endpoint. Worth building only after `CONTROLLED` works end to end from
plain file upload — don't couple the two.

---

## 13. Phasing

1. **Plumbing.** `GenRequest` type, single image becomes `{front: ...}` under
   `MULTIVIEW`. No behaviour change. Ship and confirm no regression.
2. **View preprocess + validate.** Headless, CLI-testable.
3. **Swap shape backend to `dit-v2-mv-turbo`,** front-only. Confirm n=1 parity
   against current output. **Gate everything else on this.** If parity holds,
   delete the base checkpoint here.
4. **Enable extra slots** end to end.
5. **UI:** mode toggle, slot strip, banner, per-slot progress.
6. **Preset integration:** octree resolution, steps, guidance per style preset.
   Fold in the fixed 3D-safety suffix and negative here (§6a role 1) — prompt
   composition first, LM Studio rewrite second.
6a. **LM Studio image gate.** Highest value-per-line change in this plan: a
   quality loop with no new diffusion model. Build after preset composition so
   the gate has a stable target to judge against.
6b. **Image quality techniques (§6b).** Best-of-N first — it reuses the gate you
   just built. Then high-res-and-downsample, per-preset CFG, the negative-prompt
   log. Delighting pass and framing LoRA last; both are bigger jobs.
7. **Omni, bbox only.** Second checkpoint, mode dispatch, scheduler grouping.
   Smallest possible surface to prove the `CONTROLLED` path.
8. **Remaining Omni kinds:** voxel, point cloud, skeleton.
8a. **Material controls.** Tier 1 scalars plus decoupled `MATERIAL_APPLY` /
   `EXPORT`. Can land any time after Phase 5 — it's independent of the
   shape-mode work and gives an immediate quality-of-life win.
9. **Blender control export.**

Phase 3 is the decision point for the whole plan. Phase 7 is the decision point
for Omni — if the second resident checkpoint hurts more than bbox control helps,
stop there and keep `MULTIVIEW` as the only mode.

---

## 14. Tests

Fixture set: 5 objects × 4 clean orthographic renders from Blender, with
ground-truth meshes — and for Omni, the matching bbox/voxel/point/skeleton
payloads derived from those same meshes.

- **Parity:** mv at front-only vs current base model, fixed seed. Chamfer + visual.
- **Quality curve:** 1 / 2 / 3 / 4 slots vs ground truth. Confirm improvement and
  where it flattens.
- **Omni control fidelity:** does the output actually respect the bbox aspect
  ratio? Measure output L/W/H against requested. This is the whole point of the
  feature and the easiest thing to get silently wrong.
- **Mode exclusivity:** submit views + control together; must be rejected at
  `validate()`, not at the model.
- **Adversarial:** wrong object in Back, mismatched scale, mismatched lighting,
  control signal from a different object — all caught by §5.
- **Type safety:** pass a path, an ndarray, a non-RGBA image into the adapter;
  clean rejection, never reaching OpenCV.
- **Resource:** peak VRAM across the octree ladder × slot count, both modes;
  confirm evict/load ordering and scheduler grouping hold.
- **Resume:** kill mid-preprocess and mid-generate in both modes.
- **Offline:** full run, network disabled, both checkpoint sets pre-fetched.

---

## 14a. Comparison harness (A/B tester)

A built-in mode that answers "which of these two is actually better *for this
pipeline*" instead of relying on public benchmarks that measure things ANVIL
doesn't need (text rendering, photorealism).

### Shape

```python
class CompareAxis(str, Enum):
    IMAGE = "image"     # two ImageBackends, shape stage held fixed
    SHAPE = "shape"     # two shape checkpoints/modes, images held fixed

@dataclass
class CompareRun:
    axis: CompareAxis
    a: str                      # backend/checkpoint id
    b: str
    categories: list[FixtureCategory]   # default: all three
    seeds: list[int]                    # fixed, shared across A and B
```

### The three fixture categories

Chosen because they fail in different ways:

| Category | Examples | What it exposes |
|---|---|---|
| `HARD_SURFACE` | treasure chest, throne, crate, lantern | Flat planes, sharp edges, right angles. Catches wobbly surfaces and rounded-off corners. |
| `ORGANIC` | alien creature, gnarled beast, deep-sea thing | Asymmetry, limbs, appendages. Catches fused limbs, melted detail, and back-side hallucination. |
| `COMPLEX_GEOMETRY` | interlocking rings, impossible/ambiguous solid, lattice frame | Occlusion, holes, self-intersection, depth ambiguity. Catches topology failures and filled-in negative space. |

3 prompts per category, 3 seeds each = 9 runs per candidate, 18 per comparison.
Enough signal without turning into an afternoon.

Fixture prompts and their reference images live in the repo as a versioned
fixture set, so results stay comparable across ANVIL versions.

### Fairness rules — the part that's easy to get wrong

- **Everything not under test is frozen.** Same seeds, same style preset, same
  octree resolution, same steps, same matting model.
- **`SHAPE` axis reuses cached images.** Generate the image set once, store it,
  feed the identical PNGs to both shape candidates. Regenerating per candidate
  makes the comparison meaningless.
- **`IMAGE` axis holds the shape checkpoint fixed** and reuses one loaded
  instance across both image candidates' outputs.
- **Never alternate checkpoints.** Run all of A, evict, run all of B. Interleaved
  A/B would spend most of the wall clock on model swaps (§8).

### Metrics

Automated, per run:

- VLM gate pass rate and which checks failed (§6a) — `IMAGE` axis only
- Silhouette IoU: rendered mesh front view vs the input image alpha
- Mesh health: watertight, non-manifold edge count, floating component count
- Face count after simplification
- Wall-clock time and peak VRAM

Judged:

- VLM side-by-side on turntable renders, forced choice with a stated reason
- Human side-by-side in the report — the automated metrics don't capture "looks
  right"

### Output

An HTML report in the session directory: per-category rows, A and B columns,
four-angle turntable renders, metric table, and an aggregate win/loss/tie count.
Present it as a file rather than in-app — you'll want to compare reports across
runs.

### Integration

Runs as a batch queue job type, so checkpointing, resume, and the resource
manager all apply unchanged. A comparison is just a long queue with a report
step at the end.

Build this **after Phase 4**. It's the tool that answers Phase 3's parity
question, Phase 7's is-Omni-worth-it question, and the FLUX-vs-Qwen question —
so it pays for itself three times over.

---

## 15. Risks

1. **Version drift.** A bump to 2.1 silently disables MV mode. Pin checkpoints
   and assert the expected mode flag at startup — fail loudly, not quietly.
2. **Texture expectation gap.** Users assume four views fix the back texture.
   They fix the back *shape*. Say so in the UI.
3. **Inconsistent views regress quality.** §5 is the mitigation.
4. **AI-generated filler views.** Discourage explicitly. A hallucinated back view
   becomes a hard geometric constraint — worse than letting the model infer it.
5. **Two resident checkpoints.** Omni doubles the model-management surface for a
   feature that may see light use. Phase 7 exists so you can measure that before
   committing. If mode-switching dominates queue time in practice, cut it.
6. **Skeleton convention mismatch.** Joint ordering that doesn't match the
   expected convention fails silently and confidently. Validate hard, and test
   against a known-good armature before trusting it.
7. **LM Studio as a hidden dependency.** Easy to let the gate become
   load-bearing, at which point a stopped LM Studio breaks generation entirely.
   Every role must soft-fail, and there should be a test that runs the full
   pipeline with the endpoint down.
8. **Gate retry loops burning time.** A checklist that's too strict will
   regenerate forever on a subject that can't satisfy it. Cap attempts, log which
   check failed most often, and loosen the checklist from that data rather than
   from intuition.
9. **Image backend sprawl.** Seven entries in §1a is a maintenance surface, not a
   feature. Ship two or three, use the harness to pick a winner, and delete the
   losers rather than keeping them "just in case."
10. **Harness self-deception.** An A/B that silently varies a second thing
   produces a confident wrong answer. Assert frozen parameters explicitly at run
   start and record every setting in the report, so a bad comparison is
   detectable after the fact.
11. **Silent material loss on GLB export.** Subsurface and anything without a
   glTF mapping vanishes without complaint. Warn at export time listing exactly
   what was dropped, and check your Blender exporter version's extension support
   rather than assuming the spec is implemented.
