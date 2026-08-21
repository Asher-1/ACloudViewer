# qYOLO — YOLO Object Detection, Instance Segmentation & Metric Depth Plugin

Run **Ultralytics YOLO GGUF models** (YOLOv8 + YOLO26) in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for real-time COCO-80 object detection, instance segmentation and metric (absolute) depth estimation.

## Architecture

```
GUI (YOLO dialog) ──► libAICore (yolo_capi) ──► GGML YOLO
                         ├── detect_rgb_json → detections JSON (class/score/box)
                         ├── seg_rgb        → typed detections + instance masks
                         └── depth_rgb      → metric depth map in meters + stats JSON
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| GGML YOLO engine | `core/AICore/src/tasks/yolo/` (port of ultralytics-ggml) |
| Plugin | `plugins/core/Standard/qYOLO/` |
| ggml patch | none required |

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

The dialog has four tabs: **Object Detection**, **Instance Segmentation**,
**Metric Depth**, and **Live (camera / video)**. Each task tab owns an
independent model combo filtered to that task's catalog (a detection tab never
offers a segment model and vice versa), its own thresholds, image input and
Run button; the Live tab lists all models and adapts its threshold row to the
selected model.

### Object Detection / Instance Segmentation tabs

1. Pick a **model variant** (YOLOv8 n→x: classic NMS head; YOLO26 n→x: end-to-end head; `-seg` variants: instance segmentation).
2. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads** (0 = auto) — shared by all tabs.
3. Set **Confidence** / **IoU** / **Top-K** thresholds.
4. Pick an input image from disk or the DB tree and click **Run** — the model downloads from cloudViewer_downloads on first use.

The annotated image is added to the DB tree: boxes + class/score labels (detection) or a translucent per-class mask tint plus boxes (segmentation) as `YOLO_<source>_<device>`, with full metadata (per-detection class/score/box/mask, runtime, device, model).

### Metric Depth tab

1. Pick a **depth model variant** (`yolo26n-depth`). The threshold row (Conf/IoU/Top-K) is hidden — depth models produce a metric depth map, not detections.
2. Set Device / Threads, pick an image, click **Run**.
3. The result is a turbo colormap (near = blue, far = red) with a range legend as `YOLODepth_<source>_<device>`, storing the depth map size, min/max/mean/p95 depth (meters) and valid-pixel count.

### Live (camera / video) tab

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

Benchmark source: [ultralytics-ggml](https://github.com/Asher-1/ultralytics-ggml)
(commit `8c356b7a`). The integrated parity harness lives in
`core/AICore/tests/yolo/` (`run_upstream_parity.sh` + `bench_compare.py`);
CUDA graph rows currently exceed the +5% gate — see
[`core/AICore/docs/cuda_graph_parity.md`](../../../core/AICore/docs/cuda_graph_parity.md).

### Live video latency — how to read the status line

The live tab's status line shows the **model latency** (preprocess + forward +
postprocess) and the **backend-resolved device**, e.g. `Objects: 3 | infer 34 ms
(CUDA0)` for detection, `Objects: 5 | infer 41 ms (Vulkan0)` for segmentation,
or `Depth 1920×1080 | 0.4–12.3 m | infer 41 ms (Vulkan0)` for depth. The number
does **not** include video decode, color conversion or cross-thread hops;
overlay updates may trail the display by 1–2 frames by design (busy frames are
skipped, not queued).
