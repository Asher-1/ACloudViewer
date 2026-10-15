# qRelateAnything

Scene relation inference for images, image batches, DB images, video files and
cameras. The plugin detects objects through the public AICore YOLO task, then
uses the public relation task to produce subject–predicate–object edges. The
upstream relation model requires object regions; detector time is reported
separately from relation time.

**Supported relation models:** 24 models across two backbone families (ViT and
ConvNeXt) with F16/F32/Q8_0 quantizations. Default is `relsgg-vits16plus-q8_0`
(recommended). ConvNeXt models use DINOv3 features (tiny/small/base/large) and
share the same relation tower architecture as ViT variants. All models are
downloaded from Hugging Face on first use with SHA-256 verification.

The relation model never sees a detector's labels, only boxes, so the detector
is interchangeable. The **Object detector** list offers, in order: YOLOE
prompt-free (default `yoloe-26s-seg-pf-f16`, upstream's `masks` mode), YOLOE and
YOLO-World with the 497 RA-4M categories (upstream's `boxes` mode vocabulary),
YOLO segmentation, RF-DETR (detection and segmentation) and closed-set YOLO
detectors. F16 and Q8_0 rows with a pinned digest are listed. Segmenters draw
smoothed instance masks; detected masks are display-only because upstream's
exported graph reads image and boxes only.

Upstream does not redistribute its re-parameterised detectors (they derive from
AGPL ultralytics weights). Selecting a YOLOE or YOLO-World row bakes the same
497 category names (owned by the AICore relation task and read through
`aicore_relation_object_vocabulary_*`; the list comes from upstream's
`deploy/megasg_categories.json`) through the CLIP / MobileCLIP text encoder at
load time. This reproduces the vocabulary, not the exact upstream weights. The
qYOLO World / YOLOE tabs have an **RA-4M 497** button that fills the same list.

**Options** is grouped into collapsible sections: *Objects* and *Relations*
(open), *Video tracking and smoothing* and *Advanced* (folded). Detection
confidence follows the selected detector's official value (YOLOE prompt-free
0.25, YOLO-World 0.15, others 0.20; 0.25 for the image demo) until you edit it.
Candidate relations are ranked by score x confidence(subject) x
confidence(object) as upstream does; floors still apply to the raw score.

Upstream has two operating points, and **Options → Operating point** selects
between them (editing any threshold switches it to Custom):

| Setting | Official video reel (default, `render_video.py`) | Official image demo (`demo_webcam.py`) |
|---|---|---|
| Detection confidence / NMS IoU | 0.20 / 0.5 | 0.25 / 0.5 |
| Maximum objects | 24 | 32 |
| Largest object (fraction of image) | 0.85 | no limit |
| Relation / spatial floor | 0.45 / 0.42 | 0.5 / 0.5 |
| Relations shown (spatial) | 8 (3) | 10 |
| Spatial / semantic separation | on | off |

Both use class-agnostic dedupe at IoU 0.7 and one edge per (object, predicate)
and per (subject, object, leading word). **Relation vocabulary** selects the
curated predicate list of either script (65 or 35 phrases) or the full model
vocabulary, which also contains unusual phrases such as "has holding"; typing
in **Custom predicates** switches it to Custom. The decomposed streams are
re-calibrated with the GGUF `ra.calibration_a/b`
(`aicore_relation_calibration`), because upstream's native decomposed branch
reports uncalibrated sigmoid scores. The model's internal pair budget stays 128.

### Video tracking and smoothing

Video and camera frames (not single images or batches) are stabilised with a
port of upstream `deploy/render_video.py` (`RelateAnythingTemporal.h`, Qt-free
and unit-tested): two-stage IoU tracking over Kalman-filtered boxes with
majority-vote labels, a 1-D Kalman filter on each relation's calibrated
log-odds (or EMA with **Kalman relation belief** off), on/off hysteresis, alpha
fades, and the same visible-edge rules. **Options** exposes every upstream
flag with its default: `--track_high` 0.25, `--iou_thr` 0.3, `--merge_iou`
0.75, `--max_age` 8, `--min_hits` 2, `--kf_q` 1, `--kf_r` 10, `--kf_rel_q`
0.05, `--kf_rel_r` 1, `--score_ema` 0.35, `--off_thr` 0.38, `--on_frames` 2,
`--off_frames` 6, `--max_per_pair` 2, `--topk` 24; `--on_thr`, `--spatial_on_thr`,
`--max_edges` and `--max_spatial` reuse the relation floors and caps above.
Tracks and beliefs reset when the stream restarts, seeks, stops or changes
source. The one deliberate difference is pacing: frames are consumer-driven, so
every delivered frame is inferred (upstream `--stride` 1) and `--max_age`
counts inferred frames. Upstream's Kalman box filter has zero position/velocity
cross terms, so its velocity never moves; the port reproduces that as written.

## Build

```bash
cmake -S . -B build_app -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QRELATEANYTHING=ON -DAICore_BUILD_TESTS=ON
cmake --build build_app --target QRELATEANYTHING_PLUGIN test_relate_workflow -j4
```

The CLIP tokenizer uses the existing Qt Core dependency for Unicode NFC
normalization, lowercasing and Unicode regular expressions before byte BPE.
AICore and its white-box test archive need no direct ICU targets or additional
platform package requirements. Unicode token IDs are checked against the
current upstream tokenizer.

RelateAnything inference uses ggml and one ordered patch,
`3rdparty/ggml/patches/relation_merged/0001-relateanything-ggml.patch`.
It has no ATen, CUTLASS, libtorch or PyTorch runtime dependency. The independent
SAM3D task owns its existing mem-efficient attention headers; RelateAnything
does not consume them. Upstream's process-global `ggml-runtime-options` API is
also not copied into the monolithic runtime: the relation C ABI carries the
same task-owned precision choices without environment-variable state shared by
other AICore tasks.

The default arithmetic matches the current upstream automatic policy: CUDA
models whose matrix weights are all F32 use precise arithmetic; other models
use optimized arithmetic. **Model arithmetic** offers **Automatic
(recommended)**, **Precise** and **Optimized** explicitly. CPU, CUDA and Vulkan
are supported by the task.

## Use

For a first run:

1. Open **Plugins → RelateAnything** and click **Try sample data**. Wait for
   download and verification; `horse.jpg` is the only checked image initially.
2. Keep the default relation model, automatic object detection and
   **Automatic (recommended)** arithmetic, then click **Run**. Missing relation
   and YOLO weights download into the shared cache before inference resumes.
3. Inspect the annotated preview and **Results** table. For example, a detected
   person and horse can produce a `person → riding → horse` triplet. Scores and
   available relations depend on the chosen model and detected objects.
4. Check more images for a batch, or select **Video / camera** and click **Run**
   to process the sample video. **Cancel / stop** ends the current operation.
   Use **Export JSON / CSV / text…** or **Add result image to DB** to keep results.

Open **Plugins → RelateAnything**. **Try sample data** uses the shared sample-data
button and repository, verifies the published archive, and fills the image list
with six images and the video path with the default WebM on the same click.
All six images are checkable; only the recommended `horse.jpg` is checked
initially. Check additional images to include them, or use **Check all** and
**Uncheck all**. Clicking an image previews it without changing its checkbox.
The video tab also accepts other video files and cameras. Playback advances after
inference completes, including cancellation and error paths.

Choose a relation model (24 available: 12 ViT + 12 ConvNeXt) and run. The
default object source detects objects with the selected YOLO model. To use
existing annotations, choose **Use supplied object boxes** in Options and select
a JSON file. This skips YOLO download and inference. The file accepts the
upstream plain `[[x1,y1,x2,y2], ...]` format, or an object containing `boxes`,
optional `labels`, `scores`, `image_size: [width, height]`, and `masks`. Coordinates are pixels in the original image. Supply at
least two boxes and increase **Maximum objects** when needed. Optional masks
are binary byte planes in object order, with exactly `N * height * width` bytes;
a `masks` path in the JSON resolves relative to that file. The Options mask
field overrides it.

All twelve published relation GGUFs (nine relation weights and three text
students) plus twelve ConvNeXt models are listed by the public catalog (25 total
including tokenizer). Model downloads use the shared verified downloader and
existing task cache directories. The catalog pins the published repository
revision and repairs outdated cached weights by size and SHA-256. Current Q8
exports retain the vocabulary head in F16, matching upstream. Custom predicates
are separated by `|`; they load the matching native text student and pinned
official CLIP tokenizer into an in-memory vocabulary. The three published
student GGUFs are supported through this pipeline and the public text embedding
API. For an existing vocabulary, select **Vocabulary vectors** and a matching
**Vocabulary names** JSON list. Vectors are row-major F32 `[predicate_count,
512]` values and are normalized by AICore. They replace custom predicate text
and do not require a student download. An optional predicate type map is a JSON
object mapping names to booleans (`true` for spatial, `false` for semantic).

Repeated inputs reuse the worker's loaded session. Editing a model, imported
box/mask file, vocabulary or predicate type map at the same path reloads the
session when its file size or modification time changes.

The public text embedding API retains student weights and tokenizer tables in
its context after the first call. The first call's timing includes loading;
later calls reuse those immutable assets and build the same upstream graphs.
Context teardown releases the cached data, and returned embeddings keep their
own storage.

**Decompose** returns the
upstream spatial and semantic partitions.

The result table can be filtered. JSON preserves typed object coordinates,
relation indices, scores, model/device/source and stage timings; CSV and text
export readable triplets. Video results are spooled to disk rather than retained
without limit in memory. **Add to DB** stores the annotated image and full JSON
metadata. Closing, cancelling, changing input and reopening share the persistent
worker lifecycle used by qRAM.

## Validation

```bash
export AICORE_TEST_RELATE_UPSTREAM=/path/to/pinned/RelateAnything-GGML
python3 core/AICore/scripts/validate_all.py --build build_app \
  --tasks relation --backend cuda --full \
  --output build_app/Testing/relation-cuda.json
QT_QPA_PLATFORM=offscreen ./build_app/bin/plugin_test/test_relate_workflow \
  --download-sample --real --backend cuda \
  --output-dir build_app/Testing/relate/workflow
```

The native baseline is the local upstream source snapshot recorded in
`core/AICore/src/tasks/relation/UPSTREAM.json` (revision
`62f8540a2453349b371a51154ac9b5ed69404bfa` plus the listed working-tree hashes).
The probe rejects changed sources and stale builds.
Set `AICORE_TEST_RELATE_UPSTREAM_BUILD` when selecting a native build with
matching compiler/backend options. Independent checkpoint references are
verified and reused under `build_app/Testing/framework-references/relation`.
The LIGHT tier runs 16 default-mode model/pipeline rows; `--full` adds the same
16 rows with strict arithmetic. The gate consumes every published model, compares raw outputs against that native
baseline on three pinned image/region profiles (RA4M, PSG and IndoorVG), plus
independently generated original-checkpoint PyTorch outputs, and
brackets the host forward latency with native runs. Repeated-output stability
checks run separately from the timing loop, keeping full-tensor validation scans
out of the measured forward sequence. Shared-backend locking,
output ownership, borrowed strides, cancellation, shutdown with live sessions,
repeated output stability and GUI/video workflows have focused probes. Reports
include model and binary hashes, options and raw artifacts. A gate invocation is
not evidence until its report passes.

The manifest declares absolute F32/F16/Q8 checkpoint bounds and checks the
error added by integration for every predicate and pair logit. These bounds
come from the pinned upstream machine policy, while the integration probe
uses its three catalog sources rather than the upstream 50-image G1 protocol.
Both scopes are recorded explicitly. Native parity, absolute checkpoint
accuracy and the unchanged 5% native latency gate must all pass; agreement
with a quantized native output alone does not close accuracy acceptance.
Native and candidate iteration samples, p50/p95 and CPU affinity are retained
so scheduling noise can be distinguished from a repeatable regression. See
[the acceptance ledger](TODO.md) for completed checks and remaining gates.

For CPU diagnosis, the native probe also accepts `--upstream-build` to compare
a second upstream build with matching ISA/compiler options. It still rejects
changed sources, stale binaries and the wrong backend, and records the selected
build and timed binary hash. Keep the original native build comparison as well:
different compile targets can change end-user latency even with identical graphs.
The test-only `AICORE_TEST_RELATE_UPSTREAM_BUILD` override selects that audited
build when invoking the shared matrix runner; it does not configure inference
or change production defaults.

## Provenance

Private graph/preprocessing/tokenizer code is adapted from
[RelateAnything-GGML](https://github.com/Asher-1/RelateAnything-GGML) at the revision
above. The copied upstream license is AGPL-3.0; see
`core/AICore/src/tasks/relation/LICENSE.upstream`. The tokenizer and Pillow
preprocessing notices are retained alongside the corresponding sources. See [model and data card](models/MODEL_CARD.md).
