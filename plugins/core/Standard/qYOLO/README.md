# qYOLO — YOLO Detect, Segment & Depth

Ultralytics YOLO object detection (COCO 80 classes — see full list below),
instance segmentation, and metric depth for ACloudViewer — **native C++ GGML**.

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

Twenty-one variants (10 detection: YOLOv8 n->x with the classic NMS head and
YOLO26 n->x with an end-to-end head; 10 segmentation: YOLOv8-seg n->x and
YOLO26-seg n->x; 1 metric depth: yolo26n-depth), each in
3 quantizations (f32 / f16 / q8_0) — 63 models total — on
[cloudViewer_downloads yolo_gguf_models](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/yolo_gguf_models)
are listed in the model combo. Each task tab (Object Detection / Instance
Segmentation / Metric Depth) shows only its own task's models; the Live tab
lists all of them.

## Detection Classes

All YOLO detection and segmentation models are pretrained on the **COCO
dataset** with **80 object classes** — identical to the official Ultralytics
YOLO releases. The class names are embedded in each GGUF model under the
`yolo.class_names` metadata key and are read at runtime (model metadata
default: `yolo.nc` = 80).

```
 0: person         1: bicycle        2: car            3: motorcycle
 4: airplane       5: bus            6: train          7: truck
 8: boat           9: traffic light  10: fire hydrant  11: stop sign
12: parking meter  13: bench         14: bird          15: cat
16: dog           17: horse         18: sheep         19: cow
20: elephant      21: bear          22: zebra         23: giraffe
24: backpack      25: umbrella      26: handbag       27: tie
28: suitcase      29: frisbee       30: skis          31: snowboard
32: sports ball   33: kite          34: baseball bat  35: baseball glove
36: skateboard    37: surfboard     38: tennis racket 39: bottle
40: wine glass    41: cup           42: fork          43: knife
44: spoon         45: bowl          46: banana        47: apple
48: sandwich      49: orange        50: broccoli      51: carrot
52: hot dog       53: pizza         54: donut         55: cake
56: chair         57: couch         58: potted plant  59: bed
60: dining table  61: toilet        62: tv            63: laptop
64: mouse         65: remote        66: keyboard      67: cell phone
68: microwave     69: oven          70: toaster       71: sink
72: refrigerator  73: book          74: clock         75: vase
76: scissors      77: teddy bear    78: hair drier    79: toothbrush
```

**Note:** The Metric Depth model (`yolo26n-depth`) does not produce detection
classes — it outputs a per-pixel absolute depth map in meters.

## Performance

Full benchmark charts (latency by backend, speed by dtype/model, segment
latency) are in [models/MODEL_CARD.md](models/MODEL_CARD.md).

Qualitatively:

- nano -> xlarge trades speed for recall; start with the nano variants and
  step up only when needed.
- GPU backends (CUDA / Vulkan) are roughly an order of magnitude faster than
  CPU, as with the other AICore tasks.
- f16 is the recommended quantization: half the f32 download with no
  measured recall loss.

Benchmark source: [ultralytics-ggml](https://github.com/Asher-1/ultralytics-ggml)
(commit `8c356b7a`).

## Usage

### Image tabs (Object Detection / Instance Segmentation / Metric Depth)

1. **Plugins -> YOLO Detect, Segment & Depth**
2. Pick the task tab — **Object detection**, **Instance segmentation** or
   **Metric depth**; each tab's model combo lists only that task's models
   (the task is a property of the model, not a runtime switch)
3. Select a variant (downloads on first Run if missing); detection/segment
   models take confidence / IoU / top-K, depth models hide the threshold row
4. Pick an image from disk or DB tree -> **Run**
5. An annotated ccImage is added to the DB tree: boxes + class/score labels
   for detection, a mask tint overlay for segmentation, a turbo colormap with
   a legend for depth

### Live (camera / video) tab

Play a video file or use the camera. The model combo lists all catalog models
(any task); the threshold row (Conf/IoU/Top-K) appears for detect/segment
models and hides automatically for depth models. Playback is display-paced: an
async worker infers on the latest decoded frame and the overlay (detection
boxes, segment masks, or the turbo depth blend at 65% opacity) refreshes on
completion. The capture button pushes the current annotated frame into the DB
tree.

## Outputs

- **DB tree**: `ccImage` named `YOLO_<source>_<device>` (detection/segmentation)
  or `YOLODepth_<source>_<device>` (metric depth), with metadata covering the
  per-detection class / score / box / mask, count, runtime, device and model
  filename; depth results additionally store the depth map size, the
  min / max / mean / p95 depth (meters) and the valid-pixel count under the
  `YOLO/` metadata namespace.

## References

- [ultralytics](https://github.com/ultralytics/ultralytics) (upstream PyTorch)
- [ultralytics-ggml](https://github.com/Asher-1/ultralytics-ggml) (upstream ggml)