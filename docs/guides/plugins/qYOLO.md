# qYOLO — YOLO Object Detection, Instance Segmentation & Metric Depth Plugin

<p align="center">
  <img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qYOLO/images/yolo-seg.jpg" width="49%" alt="YOLO instance segmentation">
  <img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qYOLO/images/yolo-depth.jpg" width="49%" alt="YOLO metric depth">
</p>

Run **Ultralytics YOLO GGUF models** (YOLOv8 + YOLO26) in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for real-time COCO-80 object detection, instance segmentation and metric (absolute) depth estimation.

## Architecture

```
GUI (YOLO dialog) ──► libAICore (yolo_capi) ──► GGML YOLO
                         ├── *_image(aicore_image_view) → borrowed stride-aware input
                         ├── detect/segment/pose/OBB   → typed records and masks
                         └── semantic/classify/depth   → typed maps, scores or depth
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| GGML YOLO engine | `core/AICore/src/tasks/yolo/` (port of ultralytics-ggml) |
| Plugin | `plugins/core/Standard/qYOLO/` |
| ggml patch | `3rdparty/ggml/patches/yolo_merged/` (registered by the central manifest) |

The plugin passes the native Qt image storage as a borrowed, stride-aware
`aicore_image_view`; it does not repack every frame into a tightly packed RGB
scratch buffer. JSON and path entry points remain compatibility APIs and are
not used by the interactive inference hot path.

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QYOLO=ON \
  .

cmake --build build_app --target QYOLO_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` (shared with qDA3, qDeepLSD, qFaceDetect, qLightGlue, qFreeSplatter, qRFDetr, qRMBG) |
| `PLUGIN_STANDARD_QYOLO` | This plugin |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQYOLO_PLUGIN.so`.

## GUI usage

**Menu:** Plugins → **YOLO Detect, Segment & Depth**

The dialog shows a grouped **task list** on the left with one panel per task on the right: **Object Detection**, **Instance Segmentation**, **Metric Depth**, **Pose (Keypoints)**, **Oriented Boxes**, **Classification** and **Semantic Segmentation** under *Closed-set tasks*; **Open-Vocab Detect (World)** and **Open-Vocab Segment (YOLOE)** under *Open-vocabulary*; **Live (camera / video)** under *Capture*. **Device** and **Threads** are shared controls rendered once above the task list. Each task panel owns an
independent model combo filtered to that task's catalog (a detection panel never
offers a segment model and vice versa), its own thresholds, image input and
Run button; the Live page lists all models and adapts its threshold row to the
selected model.

Every task panel has a **Try sample data** button that loads that task's default sample image from the shared `objects_detection_data` test-data cache (downloaded on first use): **Classification** loads the single-subject `cat.jpg`, **Oriented Boxes** loads the DOTA-style aerial `aerial_airport.jpg`, **Pose** loads `000000087038.jpg` (multiple people in dynamic poses), **World / YOLOE** load `party_hats.jpg` (one differently colored party hat per person — prompts like `adult with red hat` select the matching person only), and all other tasks load the COCO street scene `000000397133.jpg`.

The **World** panel also accepts the *Multilingual CLIP Bridge* text tower (`mclip-labse-vitb32-q8_0.gguf` default / `-f16.gguf`, a DistilBERT tower projected into the CLIP ViT-B/32 text space): prompts in 100+ languages — including Chinese — are mapped into the same space the detection head was trained against, so the head consumes them unchanged. Bridged prompts score ~4x lower than native-English ones; selecting the bridge therefore **automatically sets the panel confidence to 0.03 on every tower switch** (measured: Chinese max 0.046 vs English 0.186 on the sample scene — all detections at 0.03 hit the prompted subjects), and switches back to 0.25 on a native tower. The bridge is a multilingual tool: for pure-English prompts the native CLIP tower scores ~3x higher on the same prompt (party-hats scene: 0.166 vs 0.053) — the hint label calls this out while English text is entered. Note that CLIP-style heads bind attributes weakly: with `person with yellow hat` the strongest response is often the most salient person regardless of hat color (measured identical top-1 box on both towers), so short category nouns (`hat`, `red hat`) rank more reliably than long descriptive phrases. The YOLOE panel requires the MobileCLIP2-B tower (its head lives in a different embedding space) and stays English-only.

### Object Detection / Instance Segmentation panels

1. Pick a **model variant** (YOLOv8 n→x: classic NMS head; YOLO26 n→x: end-to-end head; `-seg` variants: instance segmentation).
2. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads** (0 = auto) — shared by all panels.
3. Set **Confidence** / **IoU** / **Top-K** thresholds.
4. Pick an input image from disk or the DB tree and click **Run** — the model downloads from cloudViewer_downloads on first use.

The annotated image is added to the DB tree: boxes + class/score labels (detection) or a translucent per-class mask tint plus boxes (segmentation) as `YOLO_<model>_<source>_<device>` (model tag = checkpoint filename stem, e.g. `yolov8s-world-f16`), with full metadata (per-detection class/score/box/mask, runtime, device, model).

### Metric Depth panel

1. Pick a **depth model variant** (`yolo26n-depth`). The threshold row (Conf/IoU/Top-K) is hidden — depth models produce a metric depth map, not detections.
2. Set Device / Threads, pick an image, click **Run**.
3. The result is a turbo colormap (near = blue, far = red) with a range legend as `YOLODepth_<model>_<source>_<device>`, storing the depth map size, min/max/mean/p95 depth (meters) and valid-pixel count.

### Live (camera / video) page

1. Pick any model (the combo lists all catalog tasks), start the camera or open a video file (reuses `video_base` playback: seek, speed, frame stepping).
2. The threshold row (Conf/IoU/Top-K) appears for detect/segment models and hides automatically when a depth model is selected — the layout adapts to the chosen model.
3. Playback is display-paced: an async worker infers on the latest decoded frame and the overlay refreshes on completion — detection boxes, segment masks, or the turbo depth blend at 65% opacity.
4. **Capture** stores the current annotated frame into the DB tree.

## Models

Official weights: [cloudViewer_downloads yolo_gguf_models release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models) — GGUF conversion of Ultralytics YOLO: **21 variants × 3 quantizations (f32 / f16 / q8_0) = 63 models**.

| Variant family | Variants | Task | Head |
|----------------|----------|------|------|
| YOLOv8 | n / s / m / l / x | Detection (COCO 80) | classic + NMS |
| YOLO26 | n / s / m / l / x | Detection (COCO 80) | end-to-end |
| YOLOv8-seg | n / s / m / l / x | Segmentation | classic + NMS |
| YOLO26-seg | n / s / m / l / x | Segmentation | end-to-end |
| YOLO26 depth | yolo26n-depth | Metric depth | end-to-end |

Default: `yolov8n-f16.gguf`. Model cache directory: `yolo_models/`.

### Detection Classes — COCO 80

All YOLO detection and segmentation models (YOLOv8 and YOLO26 families) are
pretrained on the **COCO dataset** with **80 object classes**, identical to the
official Ultralytics YOLO releases. The class IDs and names are embedded in
each GGUF model under the `yolo.class_names` metadata key and are read at
runtime — no hardcoded class table in the plugin.

| # | Class | # | Class | # | Class | # | Class |
|---|-------|---|-------|---|-------|---|-------|
| 0 | person | 20 | elephant | 40 | wine glass | 60 | dining table |
| 1 | bicycle | 21 | bear | 41 | cup | 61 | toilet |
| 2 | car | 22 | zebra | 42 | fork | 62 | tv |
| 3 | motorcycle | 23 | giraffe | 43 | knife | 63 | laptop |
| 4 | airplane | 24 | backpack | 44 | spoon | 64 | mouse |
| 5 | bus | 25 | umbrella | 45 | bowl | 65 | remote |
| 6 | train | 26 | handbag | 46 | banana | 66 | keyboard |
| 7 | truck | 27 | tie | 47 | apple | 67 | cell phone |
| 8 | boat | 28 | suitcase | 48 | sandwich | 68 | microwave |
| 9 | traffic light | 29 | frisbee | 49 | orange | 69 | oven |
| 10 | fire hydrant | 30 | skis | 50 | broccoli | 70 | toaster |
| 11 | stop sign | 31 | snowboard | 51 | carrot | 71 | sink |
| 12 | parking meter | 32 | sports ball | 52 | hot dog | 72 | refrigerator |
| 13 | bench | 33 | kite | 53 | pizza | 73 | book |
| 14 | bird | 34 | baseball bat | 54 | donut | 74 | clock |
| 15 | cat | 35 | baseball glove | 55 | cake | 75 | vase |
| 16 | dog | 36 | skateboard | 56 | chair | 76 | scissors |
| 17 | horse | 37 | surfboard | 57 | couch | 77 | teddy bear |
| 18 | sheep | 38 | tennis racket | 58 | potted plant | 78 | hair drier |
| 19 | cow | 39 | bottle | 59 | bed | 79 | toothbrush |

**Note:** The Metric Depth model (`yolo26n-depth`) does not produce detection
classes — it outputs a per-pixel absolute depth map only.

See [MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/models/MODEL_CARD.md) for download links and licensing.

## Backends

CUDA / Vulkan / Metal / CPU — AICore's unified device resolution (`Auto` picks
the best available GPU; CUDA → Vulkan → CPU on Linux/Windows, Metal → CPU on
macOS). Thread count is configurable.

## Performance

Full benchmark charts (latency by backend, speed by dtype/model, segment
latency) are in [MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/models/MODEL_CARD.md).

Qualitatively:

- nano → xlarge trades speed for recall; start with the nano variants and step up only when needed.
- GPU backends (CUDA / Vulkan) are roughly an order of magnitude faster than CPU, as with the other AICore tasks.
- f16 is the recommended quantization: half the f32 download with no measured recall loss.

Benchmark source: [ultralytics-ggml](https://github.com/Asher-1/ultralytics-ggml).
The integrated parity and performance harness lives in
`core/AICore/tests/yolo/`. Use `run_yolo_model_matrix.py` for model,
quantization and backend coverage, and treat skipped asset rows as incomplete
coverage rather than a pass. Current acceptance rules and controlled CUDA
build comparison are documented in
[`core/AICore/tests/TESTING.md`](../../../core/AICore/tests/TESTING.md).

### Live video latency — how to read the status line

The live tab's status line shows the **model latency** (preprocess + forward +
postprocess) and the **backend-resolved device**, e.g. `Objects: 3 | infer 34 ms
(CUDA0)` for detection, `Objects: 5 | infer 41 ms (Vulkan0)` for segmentation,
or `Depth 1920×1080 | 0.4–12.3 m | infer 41 ms (Vulkan0)` for depth. The number
does **not** include video decode, Qt signal delivery or rendering. The input
view itself is borrowed with its row stride, so the plugin does not perform a
mandatory full-frame packing copy before each AICore call.

### YOLOE prompt modes — text, visual (SAVPE), prompt-free

The YOLOE tab offers three ways to define the categories, mirroring the
official ultralytics YOLOE prompting modes:

- **Text prompt** (default, non-`-pf` checkpoints): enter comma-separated
  class names; the MobileCLIP2-B tower encodes them (downloaded
  automatically). Short category nouns score far better than descriptive
  phrases.
- **Visual prompt** (SAVPE): switch the prompt mode to *Visual prompt* and
  draw one example box per target: the mode swaps a full-width drawing
  canvas in below the config row (it re-fits with the dialog), replacing
  the small preview thumbnail. The checkpoint's
  SAVPE encoder derives one class embedding per box; results are labeled
  `object0`, `object1`, … (official semantics: example boxes group targets,
  they do not carry names). Requires a non-`-pf` YOLOE GGUF converted with
  savpe weights — regenerate via `core/AICore/src/tasks/yolo/tools/convert_yoloe_savpe_gguf.py`
  (writes `yolo.savpe = 1` + the `savpe.*` tensors); the loader validates
  the weight shapes against the actual FPN features at load. Visual prompts
  take precedence over a class list when both are present.
- **Prompt-free** (`-pf` checkpoints): no input at all — the checkpoint
  matches against its built-in 4585-entry LRPC vocabulary. `-pf` variants
  reject both a class list and visual prompts.

Leaving the YOLOE class list empty in text mode fails the Run with an
actionable hint (enter classes, draw visual prompts, or pick the `-pf`
variant of the same scale — the official no-input path), because a
fresh upstream `*-seg.pt` has no usable built-in vocabulary: without
`set_classes` the official runtime falls back to a zero embedding and emits
80 numeric placeholder classes ("0"…"79"), which is not usable recognition.

## Multi-object tracking and ReID

### Track tab

The Live tab's *Track* checkbox adds stable per-object ids on top of any
detect/segment/pose/obb model. The type combo exposes the six official
`ultralytics/cfg/trackers` backends (`tracktrack` — the official default —
plus `bytetrack`, `botsort`, `ocsort`, `deepocsort`, `fasttrack`); switching
the type resets the threshold spins and the GMC combo to that tracker's
official YAML defaults (the upstream `tracker=<yaml>` selection semantics).
Every threshold maps 1:1 to a YAML key (`track_high_thresh`,
`track_low_thresh`, `new_track_thresh`, `track_buffer`, `match_thresh`),
and enabling Track lowers Conf to the official track-mode default of 0.1
when it still carries the detect default. Overlay labels render in the
official `results.plot()` format (`id:<n> person 0.13`) with per-class
colors, and the stroke/font sizes follow the official `Annotator`
defaults for the rendered image size.

Headless use: `ACloudViewer -SILENT -YOLO_TRACK MODEL <gguf> VIDEO|FRAMES_DIR
<source> [TRACKER tracktrack] [GMC sparseOptFlow] [CONF 0.1] [REID 0|1]
TRACKS_JSON out.jsonl` (jsonl rows mirror the upstream `write_track_json_entry`
schema, including the TRACKTRACK loose-NMS `detections_del` recovery rows);
agents reach the same command through the `yolo.track` JSON-RPC method.

![qYOLO tracked output: stable ids across detection gaps and occlusion](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/images/yolo-track.jpg?raw=1)

Tracked output on a moving-target sequence (official detector detections fed
through the ported tracker): identities survive the detection gap and the
occlusion window with stable ids; the sequence is the same one the
field-by-field parity harness runs (`tests/track_parity_check.py`, official
runtime vs the C++ tracker, 6/6 trackers identical).

### ReID (appearance re-identification)

The *ReID* checkbox engages the official `with_reid` cosine term for the
BoT-SORT family (`botsort` / `deepocsort` / `tracktrack`; the other three
trackers never read it). Two encoder paths exist, mirroring the official
`trackers/utils/reid.py` `build_encoder`:

- **`model: "auto"` (default of every official YAML, fully supported)**:
  the detector itself provides the appearance features. The yolo engine
  exports the detect head's input feature levels (P3/P4/P5), pools every
  spatial position to a uniform vector (channel-group mean, exactly the
  official `get_obj_feats`), and gathers one row per kept detection
  (`aicore_yolo_features_view`). The tracker blends the rows into a
  normalized EMA state (`smooth_feature`) and folds the cosine distance
  into the association cost (`embedding_distance / 2`, gated by
  `proximity_thresh` / `appearance_thresh`, min-fused for BoT-SORT and
  Deep OC-SORT, weight-fused for TrackTrack). End2end heads have no input
  feature maps — exactly like the official auto path, which swaps in a
  separate encoder there — so those streams degrade to motion-only
  association.
- **`model: <path>` (explicit encoder)**: `aicore/reid_capi.h` wraps a
  classify-task GGUF (the official auto fallback encoder,
  e.g. `yolo26n-cls-*`) as `aicore_reid_load_opts` /
  `aicore_reid_embed_image` — crops follow the official `save_one_box`
  semantics (gain 1.02, pad 10, boundary clip), are stretched to the
  model's square input, and return the pooled pre-linear feature
  (`session_read_embed`). A stream without features degrades to
  motion-only association, mirroring the upstream "feats missing"
  behavior. Dedicated appearance encoders are published as the authoritative
  family `reid-yolo26{n,s,m,l,x}-{f32,f16,q8_0}.gguf` (15 files) in the
  [`yolo_gguf_models`](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models)
  release and on
  [Hugging Face `Asher-1/yolo-gguf`](https://huggingface.co/Asher-1/yolo-gguf);
  the catalog pins their SHA-256 and the validate-all `reid-native-models`
  rows exercise them on every backend. Upstream truth: these are converted
  directly from the official `yolo26{n,s,m,l,x}-reid.onnx` assets — every
  rebuilt network verifies against its onnx graph at cosine 1.000000 before
  writing, and the AICore CUDA runtime matches the official onnx output at
  cos=0.9999 on identical CHW input. (The earlier cls-tower family
  `reid-yolo26*-cls-*` was withdrawn: its weights were not the official
  ReID models.) Conversion:
  `cpp_ggml/scripts/convert_reid_onnx_to_gguf.py` in ultralytics-ggml.

![ReID embedding similarity: same identity stays at 1.0 across the occlusion gap, cross identity at ~0.68](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/images/yolo-reid.jpg?raw=1)

Validation: `core/AICore/tools/reid_validate.py` compares the C++ probe
(`core/AICore/tools/reid_probe.cpp`, built as the `reid_probe` target into
`build_app/bin/`) against the PyTorch and ONNX Runtime baselines on a fixed
synthetic image; the PyTorch/ORT baselines must run in their own process
(numpy and libAICore cannot share one). Upstream-truth accuracy after the
RGB32 preprocessing fix: min cosine vs PyTorch 0.9975-0.9988 across
f32/f16/q8_0 and cuda/vulkan/cpu (9/9 PASS, reports in
`build_app/Testing/validate-*.json`); batch-3 latency p50 cuda ~4.6 ms,
vulkan ~2.7 ms. The yolo validate-all gate (9/9 PASS) covers the
feature-export-off bit-stability invariant. The ReID C API itself is gated
by `test_reid_capi_contract` (no assets) and `test_reid_capi_load` (the
`reid-embed` row in `core/AICore/scripts/validation_manifest.json`,
consuming the shared `yolo26n-cls-q8_0.gguf` classify asset).
