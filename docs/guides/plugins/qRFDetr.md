# qRFDetr — RF-DETR Object Detection & Segmentation Plugin

Run **RF-DETR GGUF models** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for real-time object detection (COCO 91-class layout, 80 real classes) and optional instance segmentation, with a per-class allowlist filter.

## Architecture

```
GUI (RFDetr dialog) ──► libAICore (rfdetr_capi) ──► GGML RF-DETR
                         ├── detect_rgb_json → detections JSON (class/score/box)
                         └── detection_mask → raw per-detection masks (seg models);
                             detection_mask_png → PNG form (metadata/export only)
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| GGML RF-DETR engine | `core/AICore/src/tasks/rfdetr/` (port of [rf-detr.cpp](https://github.com/mudler/rf-detr.cpp)) |
| Plugin | `plugins/core/Standard/qRFDetr/` |
| ggml patch | `3rdparty/ggml/patches/rfdetr_merged/` (CPU sgemm broadcast fold) |

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QRFDETR=ON \
  .

cmake --build build_app --target QRFDETR_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` (shared with qDA3, qDeepLSD, qLightGlue, qFreeSplatter, qRMBG) |
| `PLUGIN_STANDARD_QRFDETR` | This plugin |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQRFDETR_PLUGIN.so`.

## GUI usage

**Menu:** Plugins → **RF-DETR Detect**

### Image tab

1. Choose a **model variant** (detection: nano → large; `-seg-` variants add instance masks).
2. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads** (0 = auto).
3. Set **Threshold** (confidence) and **Top-K** (max detections).
4. *(Optional)* Expand **Class Filter** and uncheck the classes to ignore — only checked classes are detected (see [Class Filter](#class-filter-allowlist)).
5. Pick an input image from disk or the DB tree (collapsible DB list, or select in the main DB tree).
6. Click **Run** — the model downloads from cloudViewer_downloads on first use.

The annotated image (boxes + class/score labels; segmentation models additionally tint the per-object masks) is added to the DB tree as `RFDetr_<source>_<device>` with full metadata (per-detection class/score/box, count, runtime, device, model).

### Live (camera / video) tab

1. Start the camera or open a video file (reuses `video_base` playback: seek, speed, frame stepping).
2. Playback is inference-paced: each decoded frame is displayed only after its detections have been drawn, so boxes cannot drift onto a later frame.
3. **Capture** stores the current annotated frame into the DB tree.

The Live tab mirrors the Image tab's class filter; changing it mid-playback reloads the inference context automatically.

## Class Filter (allowlist)

RF-DETR models are trained on many classes (the published weights use the COCO 91-class layout — 80 named classes plus 11 empty slots). To detect only the classes you care about:

1. Run once to load the model — the filter list fills with the model's class names (indexed by class_id).
2. Expand **Class Filter (optional)**, uncheck classes to ignore (or use **All / None**, or the search box to filter by name).
3. The status label shows `N/91 classes enabled`; Run (or continue live playback) to apply.

Filtered classes are removed by the engine's post-processing, so they never appear in the result JSON, the annotation or the DB metadata — this reduces false positives and speeds up display of busy scenes. The selection is persisted per class name and carries over across sessions and across COCO-trained model variants.

## Models

Official weights: [cloudViewer_downloads RF-DETR-GGUF release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RF-DETR-GGUF) — GGUF conversion of [Roboflow RF-DETR](https://github.com/roboflow/rf-detr): **11 variants × 4 quantizations (f32 / f16 / q8_0 / q4_K) = 44 models**.

| Variant | f16 size (approx.) | Notes |
|---------|--------------------|-------|
| nano / small / base / medium / large | 63–72 MB | Detection (`base` is default) |
| seg-nano / seg-small / seg-medium | 71–75 MB | Detection + instance segmentation |
| seg-large / seg-xlarge / seg-2xlarge | 76–82 MB | Detection + instance segmentation |

Default: `rfdetr-base-f16.gguf`. Model cache directory: `rfdetr_models/`.

### Detection Classes — COCO 91-class layout

All RF-DETR detection and segmentation models are pretrained on the **COCO
dataset**. The published weights use the **COCO 91-class layout**: 80 named
classes plus 11 empty slots (the original COCO class-id numbering), so
`num_classes` reads 91 while only 80 entries carry names — e.g. `person`=1,
`car`=3, `bus`=6, `stop sign`=13, `toothbrush`=90. The class names are embedded
in each GGUF model under the `rfdetr.class_names` metadata key and are read at
runtime (the model loader verifies `class_names.length == num_classes`) — no
hardcoded class table in the plugin.

| ID | Class | ID | Class | ID | Class | ID | Class |
|----|-------|----|-------|----|-------|----|-------|
| 1 | person | 27 | backpack | 52 | banana | 74 | mouse |
| 2 | bicycle | 28 | umbrella | 53 | apple | 75 | remote |
| 3 | car | 31 | handbag | 54 | sandwich | 76 | keyboard |
| 4 | motorcycle | 32 | tie | 55 | orange | 77 | cell phone |
| 5 | airplane | 33 | suitcase | 56 | broccoli | 78 | microwave |
| 6 | bus | 34 | frisbee | 57 | carrot | 79 | oven |
| 7 | train | 35 | skis | 58 | hot dog | 80 | toaster |
| 8 | truck | 36 | snowboard | 59 | pizza | 81 | sink |
| 9 | boat | 37 | sports ball | 60 | donut | 82 | refrigerator |
| 10 | traffic light | 38 | kite | 61 | cake | 84 | book |
| 11 | fire hydrant | 39 | baseball bat | 62 | chair | 85 | clock |
| 13 | stop sign | 40 | baseball glove | 63 | couch | 86 | vase |
| 14 | parking meter | 41 | skateboard | 64 | potted plant | 87 | scissors |
| 15 | bench | 42 | surfboard | 65 | bed | 88 | teddy bear |
| 16 | bird | 43 | tennis racket | 67 | dining table | 89 | hair drier |
| 17 | cat | 44 | bottle | 70 | toilet | 90 | toothbrush |
| 18 | dog | 46 | wine glass | 72 | tv | | |
| 19 | horse | 47 | cup | 73 | laptop | | |
| 20 | sheep | 48 | fork | 74 | mouse | | |
| 21 | cow | 49 | knife | 75 | remote | | |
| 22 | elephant | 50 | spoon | 76 | keyboard | | |
| 23 | bear | 51 | bowl | 77 | cell phone | | |
| 24 | zebra | 52 | banana | 78 | microwave | | |
| 25 | giraffe | | | | | | |

**Note:** RF-DETR is an open-vocabulary architecture that supports fine-tuning
on custom datasets. The base pretrained weights distributed via
cloudViewer_downloads use the COCO 91-class layout above; when you fine-tune on
your own dataset the class list changes accordingly.

See [MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRFDetr/models/MODEL_CARD.md) for download links and licensing.

## Backends

CUDA / Vulkan / Metal / CPU — AICore's unified device resolution (`Auto` picks
the best available GPU; CUDA → Vulkan → CPU on Linux/Windows, Metal → CPU on
macOS). Thread count is configurable.

## Performance

Upstream rf-detr.cpp benchmark (2026-08-19, Ryzen 9 5950X + RTX 3060, median
ms/image end-to-end — preprocess + forward + postprocess, image load
excluded; full 44-model matrix in
[MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRFDetr/models/MODEL_CARD.md)):

- **A GPU is worth 11–17× over CPU** (CUDA f16 ÷ CPU f16 = 0.06×–0.07×,
  Vulkan f16 ÷ CPU f16 = 0.07×–0.09×). Leave `Device` on `Auto` so a
  CUDA/Vulkan GPU is picked whenever one is present.
- **Latency ladder** (fastest config per variant): nano 13 ms, base 27 ms,
  medium 27 ms, large 39 ms; seg variants 36–197 ms (seg-2xlarge peaks on
  Vulkan f16).
- **f16 — the default quantization — is also the best CPU pick**: fastest
  CPU variant (137.2 ms vs 142.8 f32 / 148.0 q8_0 at T=8), 1.86× smaller
  than f32, and 56/56 detections match f32 at IoU ≥ 0.95.
- **q4_K is the only quantization with measured accuracy loss** (80–92%
  recall vs f32; f16 and q8_0 stay at 100%) — choose q8_0, not q4_K, when you
  need a smaller download.
- **Threads**: keep at or below the physical core count. On the 16C/32T test
  CPU, base f32 is fastest at T=12 (377 ms), f16/q8_0 at T=24 (312/247 ms);
  T=32 (SMT) collapses f32 to 5520 ms. `0 = auto` therefore resolves to the
  physical-core estimate (`hardware_concurrency() / 2`), never the SMT logical
  count — measured on the same machine, base f16 CPU: T=16 → 318 ms vs
  T=32 → 5.8 s (an 18× collapse).

### Live video latency — how to read `infer` and `e2e`

The live tab's status line shows **two latencies** and the backend-resolved
device, e.g. `Objects: 3 | infer 34 ms / e2e 52 ms (CUDA0)`:

- **`infer`** = model latency (preprocess + forward + postprocess inside
  `aicore_rfdetr_detect_rgb_json`) — the same scope the upstream benchmark
  measures. This is the number to compare against the table above.
- **`e2e`** = submit→complete wall clock — includes queued-connection hops
  (GUI→worker→GUI) and GUI-thread congestion. A large gap between `infer`
  and `e2e` signals pipeline stalls (display ticks blocking the GUI thread),
  not model slowness.

Interpreting the two numbers:

- If the device shows `cpu` while you expected a GPU, the GPU lease was not
  available — the log prints `Inference device: cpu` at stream start. CPU
  latencies of hundreds of ms per frame are normal for base/seg models
  (see the table above); a GPU brings them to the 13–70 ms range.
- Verified locally (RTX 3060, base f16, 1920×1080 input): `CUDA0` 33 ms/frame
  via the exact plugin code path; CPU with auto threads (16) 318 ms/frame.
- On GPU, `infer` should be stable at ~33 ms regardless of the Threads setting —
  the GPU does the heavy lifting; CPU threads only affect the small set of
  ops that fall back to CPU through the `[gpu, cpu]` scheduler. If `infer` is
  unexpectedly high on GPU, check the device name in parentheses — a silent
  CPU fallback (GPU lease failure) is the most common cause.
- On CPU, `infer` is highly sensitive to the thread count: auto (0) resolves
  to `hardware_concurrency() / 2` (physical-core estimate) to avoid the SMT
  oversubscription collapse. See the thread table above for details.
- Neither number includes video decode or BGR→RGB conversion; frame-to-frame
  overlay updates may still trail the display by 1–2 frames by design (busy
  frames are skipped, not queued).
