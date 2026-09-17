# qGKD — General Keypoint Detection (GKDT)

General (open-world) keypoint detection on still and DB images with the
**GKDT-L** transformer (ECCV 2026), running through the in-tree ggml runtime
in `core/AICore/src/tasks/gkd/` — no Python at runtime.

![qGKD](images/qGKD.svg)

## Prompt modes

| Mode | Inputs | Result |
|---|---|---|
| **Text** | keypoint texts ("nose", "left eye", …) | one keypoint per text |
| **Visual (1-shot)** | support image + support keypoints (pixel coords) | one keypoint per support keypoint |
| **Multimodal** | keypoint texts **and** support image + keypoints (counts must match) | text row *i* fuses with visual row *i* (official semantics) |
| **Multi-object** | object classes + keypoint texts (+ optional support) | YOLO-World boxes (existing `aicore_yolo_*` WORLD models) → GKD per box |

## Dialog layout

The dialog follows the qYOLO task-page layout: a grouped mode list on the
left drives one panel per mode on the right (Text Prompts / Visual (1-shot)
/ Multimodal under *Prompt modes*, Multi-object (YOLO-World) under
*Composition*). The GKD model and Device / Threads are global controls
rendered once above the list — the same GGUF serves every mode. Each panel
keeps its own prompt values, query image and preview, and the splitter
drag state is persisted.

Optional ROI bbox (`x1 y1 x2 y2`, empty = whole image; Text panel)
restricts the query region; keypoints are rendered with a per-keypoint
score threshold (Min score, per panel).

## Model management

The model combo lists the published catalog from AICore
(`aicore_gkd_model_*`; default = **GKDT-L (Q4_K), 483 MiB**; the deprecated
upstream q4_0 build is not cataloged). Missing models are downloaded
automatically from
[Hugging Face Asher-1/GKD_GGUF](https://huggingface.co/Asher-1/GKD_GGUF) and
SHA-256 verified at ingestion (digests pinned in
`core/AICore/include/aicore/asset_digests.h`). Cache:
`~/cloudViewer_data/extract/gkd_models/`. Custom GGUF paths are also accepted.

## Sample data

Every mode panel has its own **Try sample data** button that one-click
fills ALL fields of the active mode with a bundled demo scenario (shared
`ecvTestDataRepository` flow; downloads are SHA-256 pinned and cached
under `~/cloudViewer_data/download/`). **Repeated clicks cycle through
the mode's scenario list**, and each list starts with the recommended
first-frame image from the upstream
[General-Keypoint-Detection-GGML](https://github.com/Asher-1/General-Keypoint-Detection-GGML)
README demos — together the four lists cover every image in the bundle:

- **Text** (5 scenarios): `2007_007524.jpg` + the 5 face-keypoint texts
  (official demo), then hand X-ray (`3144.png`, the official
  `hand_xray` 24-point schema — 24/24 kept at 0.9+ where the generic
  "fingertip" prompt scored 0.04),
  penguin (`adeliepenguin_107.jpg`, the official `penguin` retrieval
  hits the `animalweb` 9-point face schema — 9/9 kept at 0.54–0.91),
  chair
  (`00000016.jpg`, the official `keypoint 1..10 of chair` schema —
  in-training phrases scoring 0.83–0.93 on all ten points where
  self-invented part names kept 2/4 at ≤0.27), tiger (`000002.jpg`,
  the official `awa_pose` 39-point body schema — 24/39 kept on the
  walking side profile, paws/legs at 0.85–0.90; the official
  invisible-keypoint semantics drop unseen points).
- **Visual (1-shot)** (4 scenarios): the fixed support image
  `2007_003778.jpg` with its three annotated keypoints (left eye
  343,166 · right eye 281,158 · nose 311,197 — the only officially
  annotated image) drives the detection on rotating query images:
  `2008_000808.jpg` (front-facing pug, recommended first frame),
  tiger `000002.jpg`, penguin `adeliepenguin_107.jpg` —
  cross-instance, open-world few-shot — then the official Example 4
  cross-object variant (support = the alpaca face 615,495 ·
  483,493 · 521,549, query = the cat pair). Single-object modes
  rotate single-subject queries only; the multi-target `cat_dog.jpg`
  lineup lives in the multi-object rotation.
- **Multimodal** (2 scenarios): the official cat pair first (texts
  `left eye, right eye, nose` fuse with the support rows; counts
  satisfy the backend's `n_kps_texts == n_support_kps` contract), then
  the pug query rotation with the same fused support.
- **Multi-object** (7 scenarios, YOLO-World boxes → GKD per box): the
  official `alpaca_150.jpg` 20-point quadruped demo (classes `alpaca`),
  then bronze statues (`000000011511.jpg`, classes `person`, the
  official coco full-body schema — shoulders/elbows/wrists/hips/knees/
  ankles measure 0.41–0.94 on the sculptures), pig farm
  (`pigs_stock_farming.jpg`, classes `pigs` — the plural embeds with
  higher similarity than `pig` and doubles the kept boxes; keypoint
  texts follow the official `awa_pose` 39-point schema, 22/39 kept
  including legs/paws at 0.72–0.86; antler rows trimmed from the prompt
  list (pigs have no antlers) and scene conf 0.15 — the 0.15 band
  measured all-real and recovers the two sub-0.25 boxes), fish school
  (`fish_swim.jpg`, classes `fish`, eye/tail/fin, scene conf 0.10 —
  25 → 46 boxes, the low band rendered all-real fish), the
  `cat_dog.jpg` lineup (classes `cat, dog`, the official `animal_pose`
  20-point schema — 15/20 kept per box, the same set as the alpaca
  demo), traffic intersection
  (`car_penn2_0_1931.jpg`, classes `car`, the official carfusion
  `car keypoint 1..14` texts — GKDT is CarFusion-trained and scores
  them 0.71–0.94 where self-invented part names sit at 0.03–0.82),
  bird row
  (`pet_birds.jpg`, classes `bird`, the nabird 11-point schema with a
  self-defined bill–crown–nape–back–tail–wings skeleton). The multi panel
  also carries a **Text encoder** row (the CLIP/MobileCLIP tower that
  encodes the class names, same catalog as qYOLO); text-conditioned
  WORLD detectors reject the run without it. The bundled egocentric
  dish-washing scene (`wash_dishes_egocentric.jpg` / `human_hand`) was
  removed from the rotation: the YOLO-World CLIP text tower embeds the
  class name too weakly for reliable recall (top boxes 0.31/0.29; the
  upstream scene relies on the stronger LocateAnything detector, which
  has no GGUF asset), so the preset usually detected nothing.

A custom image can always be picked with Browse. The ROI row accepts
multiple boxes (4+ coordinates = 2+ xyxy boxes): they run as ONE batched
GKD forward over all ROIs (official `--bbox_on_input_im` semantics,
labels `object1…N`), exactly like the multi-object mode which runs its
detected boxes as a single N-box batch instead of per-box forwards.

The Output group carries an **Export COCO JSON…** button that writes the
run as an official-style COCO prediction file (categories with keypoint
names + 1-based skeleton, images, xywh boxes with the detector scores,
and x,y,score triplets; displayed keypoints only).

Skeleton rendering: scenarios with an upstream skeleton (the coco /
animal_pose / carfusion / onehand10k / keypoint5 / animalweb schemas
and the README `--skeleton` flags) draw the official bone connections
between the shown keypoints (adaptive line width, 0.4% of the image
width — half the upstream 0.8% default, which read as heavy bars on
large images). The awa_pose, hand_xray and nabird schemas leave
`skeleton` empty upstream, so the tiger/pigs and hand-X-ray and bird
presets carry self-defined topologies over the official point sets
(per-finger chains, a jaw–ear–neck–spine–tail–leg layout, and a
bill–eye–crown / nape–back–tail bird layout); bones only draw between
keypoints that pass the display cut, so undetected endpoints simply
drop out. Fish resolves through the upstream substring fallback and
has no schema skeleton, so it renders points only. Manual runs never
draw bones.

Run latency: the GKD model (and the multi-object YOLO-World + CLIP
text towers) stay resident across runs — the first Run pays the model
load, every later click skips it (the log prints "Reusing loaded
model" instead of "Model loaded"). Switching the model, device,
thread count, or — in multi-object mode — the class list / confidence
cut triggers exactly one reload. The resident contexts are released on
`QCoreApplication::aboutToQuit`, while the CUDA runtime is still alive
(freeing them at exit-time library finalization aborted the process in
the ggml CUDA buffer cleanup).

Label rendering: per-keypoint "prompt score" labels default **off** —
in multi-object scenes the label backgrounds alone cover the objects.
The Output group's **Show keypoint labels** checkbox opts in per run
(persisted in QSettings); labeled runs spread overlapping labels apart
with force-directed placement (the algorithm behind roboflow
supervision's `LabelAnnotator.smart_position`: IoU-weighted repulsion
until no overlap or an iteration cap, then labels are snapped fully
onto the canvas) and connect each moved label to its keypoint with a
thin leader arrow.

Multi-object mode reuses the YOLO-World detector models from the existing
yolo task catalog — no second model table. The default detector is
**yolov8l-world** (measured duplicate-free with calibrated confidences on
every bundled scene; s-world emits near-tie duplicate boxes once the
cut drops), and the class-aware NMS runs at IoU 0.6 so near-duplicate
wide boxes cannot become a second overlapping keypoint set. Full-family
benchmark on the demo scenes (RTX 3060, CUDA): s/m/l/x infer ~9/16/26/39 ms
per image — negligible next to the per-box GKD pass; prefer s-world for
speed, l-world for margin, x-world for the best fish/pig recall. Avoid
m-world at 0.25 (misses the pigs and dish-washing scenes entirely) and
worldv2 (uncataloged: its recalibrated vocabulary scores whole scenes
below the cut — pigs vanishes at s/m/l sizes); the published WORLD catalog
carries no n-size entry. Model loads pre-flight the device: when the free
VRAM cannot hold the weights plus compute headroom, the load fails with an
actionable message instead of a fatal backend abort, and this plugin
retries once on CPU.

## Outputs

- **DB**: rendered image with keypoints (prompt labels + scores) and boxes;
  metadata records mode, keypoint/object counts, per-stage timings, device,
  and model (`GKD/*` keys, name prefix `GKD_`).
- **PNG export** to a user directory.

## Build

```bash
cmake -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QGKD=ON ..
cmake --build . --target QGKD_PLUGIN -j4
```

Requires `AICore_ENABLED=ON` (the plugin is skipped with a warning
otherwise).

## Validation

```bash
# One-click AICore gate (light tier downloads + runs gkd_fullset-q4_K.gguf)
cmake --build build_app --target aicore-validate-all -j1
# Complete quantization matrix (all five GGUF files)
python3 core/AICore/scripts/validate_all.py --build build_app --backend cuda --full
```

## License

GKDT weights and source are for **academic research and educational use
only** (ECCV 2026; commercial use prohibited) — see
[models/MODEL_CARD.md](models/MODEL_CARD.md).
