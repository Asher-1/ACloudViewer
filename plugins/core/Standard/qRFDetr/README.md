# qRFDetr

RF-DETR real-time object detection / segmentation (COCO 80 classes) for
ACloudViewer — **native C++ GGML**.

**User guide:** [docs/guides/plugins/qRFDetr.md](../../../docs/guides/plugins/qRFDetr.md)

```
Image/Video → AICore RF-DETR GGML → detections JSON + masks → annotated ccImage → DB tree
```

## Build

```bash
cmake -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QRFDETR=ON \
  ..
make -j4 QRFDETR_PLUGIN
```

RF-DETR GGML sources live in `core/AICore/src/tasks/rfdetr/` (in-tree port of
[rf-detr.cpp](https://github.com/mudler/rf-detr.cpp)). The CPU sgemm broadcast
fold optimization is applied as a ggml patch
(`3rdparty/ggml/patches/rfdetr_merged/`).

### Unit tests (pure helpers)

JSON envelope parsing, palette, segmentation-name detection and catalog mirror
(no GGUF model required):

```bash
cmake -DBUILD_GUI=ON -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QRFDETR=ON \
  -DBUILD_UNIT_TESTS=ON ..
cmake --build build_app --target test_qrfdetr_helpers -j4
ctest -R test_qrfdetr_helpers --output-on-failure
```

## Models

See [models/MODEL_CARD.md](models/MODEL_CARD.md). Default download:

[`rfdetr-base-f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/RF-DETR-GGUF/rfdetr-base-f16.gguf)

Eight variants (nano → large, detection + segmentation) on
[cloudViewer_downloads RF-DETR-GGUF](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RF-DETR-GGUF)
are listed in the model combo.

## Usage

### Image tab

1. **Plugins → RF-DETR Detect**
2. Select model variant (downloads on first Run if missing)
3. Set device / threads / confidence threshold / top-K
4. Pick an image from disk or DB tree → **Run**
5. Annotated ccImage (boxes + class/score labels, mask tint for segmentation
   models) is added to the DB tree

### Live (camera / video) tab

Play a video file or use the camera; inference is throttled (every 5th video
frame) and detections are overlaid live. Snapshot the current annotated frame
into the DB tree with the capture button.

## Outputs

- **DB tree**: `ccImage` named `RFDetr_<source>_<device>`, metadata includes
  per-detection class / score / box, count, runtime, device and model filename.
- Segmentation models store the per-detection PNG masks inside the metadata
  (`RFDetr/DetN/…`) as well.
