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

Eleven variants (5 detection: nano → large; 6 segmentation: seg-nano →
seg-2xlarge), each in 4 quantizations (f32 / f16 / q8_0 / q4_K) — 44 models
total — on
[cloudViewer_downloads RF-DETR-GGUF](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/RF-DETR-GGUF)
are listed in the model combo.

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