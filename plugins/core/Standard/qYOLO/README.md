# qYOLO — YOLO Detect & Depth

Ultralytics YOLO object detection (COCO 80 classes) and metric depth for
ACloudViewer — **native C++ GGML**.

**User guide:** [docs/guides/plugins/qYOLO.md](../../../docs/guides/plugins/qYOLO.md)

```
Image/Video → AICore YOLO GGML → detections JSON / metric depth map → annotated ccImage → DB tree
```

## Build

```bash
cmake -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QYOLO=ON \
  ..
make -j4 QYOLO_PLUGIN
```

YOLO GGML sources live in `core/AICore/src/tasks/yolo/` (in-tree port of
ultralytics-ggml; no ggml patches are required).

### Unit tests (pure helpers)

JSON envelope parsing, palette, depth-name detection, catalog mirror and the
turbo depth colorization (no GGUF model required):

```bash
cmake -DBUILD_GUI=ON -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QYOLO=ON \
  -DBUILD_UNIT_TESTS=ON ..
cmake --build build_app --target test_qyolo_helpers -j4
ctest -R test_qyolo_helpers --output-on-failure
```

## Models

See [models/MODEL_CARD.md](models/MODEL_CARD.md). Recommended default:

[`yolov8n-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/yolo_gguf_models/yolov8n-f16.gguf)

Eleven variants (10 detection: YOLOv8 n→x with the classic NMS head and
YOLO26 n→x with an end-to-end head; 1 metric depth: yolo26n-depth), each in
3 quantizations (f32 / f16 / q8_0) — 33 models total — on
[cloudViewer_downloads yolo_gguf_models](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models)
are listed in the model combo, filtered by the selected task.

## Performance

No benchmark matrix has been measured for the GGML port yet (the
[RF-DETR card](../qRFDetr/models/MODEL_CARD.md) documents the same-harness
reference numbers). Qualitatively:

- nano → xlarge trades speed for recall; start with the nano variants and
  step up only when needed.
- GPU backends (CUDA / Vulkan) are roughly an order of magnitude faster than
  CPU, as with the other AICore tasks.
- f16 is the recommended quantization: half the f32 download with no
  measured recall loss in the sibling catalogs.

## Usage

### Image tab

1. **Plugins → YOLO Detect & Depth**
2. Pick the task — **Object detection** or **Metric depth**; the model combo
   filters accordingly (the task is a property of the model, not a runtime
   switch)
3. Select a variant (downloads on first Run if missing); detection models
   take confidence / IoU / top-K, depth models ignore them (the controls
   are disabled)
4. Pick an image from disk or DB tree → **Run**
5. An annotated ccImage is added to the DB tree: boxes + class/score labels
   for detection, a turbo colormap (near = blue, far = red) with a legend
   for depth

### Live (camera / video) tab

Play a video file or use the camera. Playback is display-paced: an async
worker infers on the latest decoded frame and the overlay (detection boxes,
or the turbo depth blend at 65% opacity) refreshes on completion. The
capture button pushes the current annotated frame into the DB tree.

## Outputs

- **DB tree**: `ccImage` named `YOLO_<source>_<device>` (detection) or
  `YOLODepth_<source>_<device>` (metric depth), with metadata covering the
  per-detection class / score / box, count, runtime, device and model
  filename; depth results additionally store the depth map size, the
  min / max / mean / p95 depth (meters) and the valid-pixel count under the
  `YOLO/` metadata namespace.

## References

- [ultralytics](https://github.com/ultralytics/ultralytics) (upstream PyTorch)
- [ultralytics-ggml](https://github.com/Asher-1/ultralytics-ggml) (upstream ggml)