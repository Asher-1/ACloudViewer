# qYOLO — YOLO Object Detection & Metric Depth Plugin

Run **Ultralytics YOLO GGUF models** (YOLOv8 + YOLO26) in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for real-time COCO-80 object detection and metric (absolute) depth estimation.

## Architecture

```
GUI (YOLO dialog) ──► libAICore (yolo_capi) ──► GGML YOLO
                         ├── detect_rgb_json → detections JSON (class/score/box)
                         └── depth_rgb → metric depth map in meters + stats JSON
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

**Menu:** Plugins → **YOLO Detect & Depth**

### Image tab

1. Pick the **task** — Object detection or Metric depth. The task is a property of the model, so the model combo just filters accordingly (it is not a runtime switch).
2. Choose a **model variant** (YOLOv8 n→x: classic NMS head; YOLO26 n→x: end-to-end head; yolo26n-depth: metric depth).
3. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads** (0 = auto).
4. Detection models take **Confidence** / **IoU** / **Top-K**; depth models ignore them (the controls are disabled).
5. Pick an input image from disk or the DB tree and click **Run** — the model downloads from cloudViewer_downloads on first use.

The annotated image is added to the DB tree: boxes + class/score labels as `YOLO_<source>_<device>` for detection, or a turbo colormap (near = blue, far = red) with a range legend as `YOLODepth_<source>_<device>` for depth, with full metadata (per-detection class/score/box or depth statistics, runtime, device, model).

### Live (camera / video) tab

1. Start the camera or open a video file (reuses `video_base` playback: seek, speed, frame stepping).
2. Playback is display-paced: an async worker infers on the latest decoded frame and the overlay refreshes on completion — detection boxes, or the turbo depth blend at 65% opacity.
3. **Capture** stores the current annotated frame into the DB tree.

## Models

Official weights: [cloudViewer_downloads yolo_gguf_models release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models) — GGUF conversion of Ultralytics YOLO: **11 variants × 3 quantizations (f32 / f16 / q8_0) = 33 models**.

| Variant family | Variants | Task | Head |
|----------------|----------|------|------|
| YOLOv8 | n / s / m / l / x | Detection (COCO 80) | classic + NMS |
| YOLO26 | n / s / m / l / x | Detection (COCO 80) | end-to-end |
| YOLO26 depth | yolo26n-depth | Metric depth | end-to-end |

Default: `yolov8n-f16.gguf`. Model cache directory: `yolo_models/`.

See [MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/models/MODEL_CARD.md) for download links and licensing.

## Backends

CUDA / Vulkan / Metal / CPU — AICore's unified device resolution (`Auto` picks
the best available GPU; CUDA → Vulkan → CPU on Linux/Windows, Metal → CPU on
macOS). Thread count is configurable.

## Performance

No benchmark matrix has been measured for the GGML port yet — see
[MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qYOLO/models/MODEL_CARD.md)
once benchmarks land. Qualitatively: start with the nano variants; a GPU is
roughly an order of magnitude faster than CPU (see the RF-DETR card for the
same-harness reference numbers); f16 is the recommended quantization.

### Live video latency — how to read the status line

The live tab's status line shows the **model latency** (preprocess + forward +
postprocess) and the **backend-resolved device**, e.g. `Objects: 3 | infer 34 ms
(CUDA0)` for detection or `Depth 1920×1080 | 0.4–12.3 m | infer 41 ms (Vulkan0)`
for depth. The number does **not** include video decode, color conversion or
cross-thread hops; overlay updates may trail the display by 1–2 frames by
design (busy frames are skipped, not queued).
