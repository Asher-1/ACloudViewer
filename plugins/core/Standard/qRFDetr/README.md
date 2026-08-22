# qRFDetr

RF-DETR real-time object detection / segmentation (COCO 80 classes — see full
list below) for ACloudViewer — **native C++ GGML**.

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

Eleven variants (5 detection: nano → large; 6 segmentation: seg-nano →
seg-2xlarge), each in 4 quantizations (f32 / f16 / q8_0 / q4_K) — 44 models
total — on
[cloudViewer_downloads RF-DETR-GGUF](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RF-DETR-GGUF)
are listed in the model combo.

## Detection Classes

All RF-DETR detection and segmentation models are pretrained on the **COCO
dataset** with **80 object classes** — identical to the official Roboflow
RF-DETR releases. The class names are embedded in each GGUF model under the
`rfdetr.class_names` metadata key and are read at runtime (the model loader
verifies `class_names.length == num_classes`).

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

**Note:** RF-DETR is an open-vocabulary architecture that supports fine-tuning
on custom datasets. The base pretrained weights distributed via
cloudViewer_downloads use the standard COCO 80 classes; when you fine-tune on
your own dataset the class list changes accordingly.

## Performance

Median latency per image (upstream rf-detr.cpp benchmark, 2026-08-19,
Ryzen 9 5950X + RTX 3060, end-to-end; full 44-model matrix in
[MODEL_CARD.md](models/MODEL_CARD.md)):

- nano 13 ms → base 27 ms → large 39 ms on CUDA (fastest config per variant);
  seg variants 36–197 ms (seg-2xlarge peaks on Vulkan f16).
- GPU backends run 11–17× faster than CPU (CUDA f16 ÷ CPU f16 =
  0.06×–0.07×).
- f16 (the default) is the fastest CPU variant: 137.2 ms vs 142.8 (f32) /
  148.0 (q8_0) at T=8, 1.86× smaller than f32, and 56/56 detections match
  f32 at IoU ≥ 0.95.
- q4_K is the only quantization with measured recall loss (80–92%);
  f16 and q8_0 stay at 100%.

## Usage

### Image tab

1. **Plugins → RF-DETR Detect**
2. Select model variant (downloads on first Run if missing)
3. Set device / threads / confidence threshold / top-K
4. Pick an image from disk or DB tree → **Run**
5. Annotated ccImage (boxes + class/score labels, mask tint for segmentation
   models) is added to the DB tree

### Live (camera / video) tab

Play a video file or use the camera. Playback is inference-paced: each decoded
frame is displayed only after its detections have been drawn, so boxes cannot
drift onto a later frame. Snapshot the current annotated frame into the DB tree
with the capture button.

## Outputs

- **DB tree**: `ccImage` named `RFDetr_<source>_<device>`, metadata includes
  per-detection class / score / box, count, runtime, device and model filename.
- Segmentation models store the per-detection PNG masks inside the metadata
  (`RFDetr/DetN/…`) as well.

## References

- [Rf-Detr-GGML](https://github.com/Asher-1/rf-detr-ggml) (upstream ggml)