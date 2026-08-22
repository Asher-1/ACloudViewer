# qRFDetr

RF-DETR real-time object detection / segmentation (COCO 91-class layout — 80
named classes, see full list below) for ACloudViewer — **native C++ GGML**.

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
dataset**. The published weights use the **COCO 91-class layout** — 80 real
classes plus 11 empty slots in the original COCO class-id numbering — so
`num_classes` reports 91 while only 80 entries carry names (e.g. `person`,
`car`). The class names are embedded in each GGUF model under the
`rfdetr.class_names` metadata key and are read at runtime (the model loader
verifies `class_names.length == num_classes`).

```
 1: person         2: bicycle        3: car            4: motorcycle
 5: airplane       6: bus            7: train          8: truck
 9: boat          10: traffic light  11: fire hydrant  13: stop sign
14: parking meter  15: bench         16: bird          17: cat
18: dog           19: horse         20: sheep         21: cow
22: elephant      23: bear          24: zebra         25: giraffe
27: backpack      28: umbrella      31: handbag       32: tie
33: suitcase      34: frisbee       35: skis          36: snowboard
37: sports ball   38: kite          39: baseball bat  40: baseball glove
41: skateboard    42: surfboard     43: tennis racket 44: bottle
46: wine glass    47: cup           48: fork          49: knife
50: spoon         51: bowl          52: banana        53: apple
54: sandwich      55: orange        56: broccoli      57: carrot
58: hot dog       59: pizza         60: donut         61: cake
62: chair         63: couch         64: potted plant  65: bed
67: dining table  70: toilet        72: tv            73: laptop
74: mouse         75: remote        76: keyboard      77: cell phone
78: microwave     79: oven          80: toaster       81: sink
82: refrigerator  84: book          85: clock         86: vase
87: scissors      88: teddy bear    89: hair drier    90: toothbrush
```

**Note:** RF-DETR is an open-vocabulary architecture that supports fine-tuning
on custom datasets. The base pretrained weights distributed via
cloudViewer_downloads use the COCO 91-class layout above; when you fine-tune on
your own dataset the class list changes accordingly.

## Class Filter (allowlist)

Only detect the classes you care about: expand **Class Filter (optional)** in
the Image tab, uncheck the classes to ignore, then Run. Filtered classes are
removed by the engine's post-processing, so they never appear in the result
JSON, the annotation or the DB metadata — this also reduces false positives
and speeds up display of busy scenes.

- The list is filled automatically after the first run with the loaded
  model's class names (indexed by class_id).
- **All / None** buttons toggle the whole list; the search box filters by name
  (check states are preserved while filtering).
- The status label shows `N/91 classes enabled`; the selection is persisted
  per class name, so it carries over across sessions and across COCO-trained
  model variants.
- The Live tab mirrors the dialog's filter; changing it mid-playback reloads
  the inference context automatically.

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