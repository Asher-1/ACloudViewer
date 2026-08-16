# qRFDetr — RF-DETR Object Detection & Segmentation Plugin

Run **RF-DETR GGUF models** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for real-time COCO-80 object detection and optional instance segmentation.

## Architecture

```
GUI (RFDetr dialog) ──► libAICore (rfdetr_capi) ──► GGML RF-DETR
                         ├── detect_rgb_json → detections JSON (class/score/box)
                         └── detection_mask_png → per-detection PNG masks (seg models)
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

1. Choose a **model variant** (nano → large; `-seg-` variants add instance masks).
2. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads** (0 = auto).
3. Set **Threshold** (confidence) and **Top-K** (max detections).
4. Pick an input image from disk or the DB tree (collapsible DB list, or select in the main DB tree).
5. Click **Run** — the model downloads from cloudViewer_downloads on first use.

The annotated image (boxes + class/score labels; segmentation models additionally tint the per-object masks) is added to the DB tree as `RFDetr_<source>_<device>` with full metadata (per-detection class/score/box, count, runtime, device, model).

### Live (camera / video) tab

1. Start the camera or open a video file (reuses `video_base` playback: seek, speed, frame stepping).
2. Inference is throttled to every 5th video frame; detections are overlaid in real time.
3. **Capture** stores the current annotated frame into the DB tree.

## Models

Official weights: [cloudViewer_downloads RF-DETR-GGUF release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RF-DETR-GGUF) — F16 GGUF conversion of [Roboflow RF-DETR](https://github.com/roboflow/rf-detr).

| Variant | Size (approx.) | Notes |
|---------|----------------|-------|
| nano / small / base / medium / large | 65–475 MB | Detection |
| seg-nano / seg-small / seg-medium | 85–330 MB | Detection + instance segmentation |

Default: `rfdetr-base-f16.gguf`. Model cache directory: `rfdetr_models/`.

See [MODEL_CARD.md](../../../plugins/core/Standard/qRFDetr/models/MODEL_CARD.md) for download links and licensing.

## Backends

CUDA / Vulkan / Metal / CPU — AICore's unified device resolution (`Auto` picks
the best available GPU; CUDA → Vulkan → CPU on Linux/Windows, Metal → CPU on
macOS). Thread count is configurable.
