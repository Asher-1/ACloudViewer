# qSAM3 — SAM 2 / 2.1 / 3 Image & Video Segmentation

Segment Anything 2 / 2.1 / 3 interactive segmentation for ACloudViewer —
**native C++ GGML** (CPU / CUDA / Vulkan / Metal, no Python or PyTorch).

```
Image + prompts → AICore SAM3 GGML → segmentation masks / detections → annotated ccImage → DB tree
```

The engine is an in-tree port of
[sam3-ggml](https://github.com/Asher-1/sam3-ggml) living in
`core/AICore/src/tasks/sam3/` (single-file C++ library running the GGUF
models on the repository-pinned ggml runtime). The required ggml operators are applied through the
standard patch chain (`3rdparty/ggml/patches/sam3_merged/`,
`igemm_fix/`).

**User guide:** [docs/guides/plugins/qSAM3.md](../../../../docs/guides/plugins/qSAM3.md)

## Build

```bash
cmake -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QSAM3=ON \
  ..
make -j4 QSAM3_PLUGIN
```

### Contract tests

```bash
cmake -DBUILD_GUI=ON -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QSAM3=ON \
  -DAICore_BUILD_TESTS=ON -DBUILD_UNIT_TESTS=ON ..
cmake --build build_app --target test_sam3_capi_contract test_sam3_contract -j4
ctest -R 'test_sam3' --output-on-failure
```

The C-API contract test (`core/AICore/tests/sam3/test_sam3_capi_contract.cpp`)
covers the full lifecycle (options / load / encode / PCS / PVS / tracker /
catalog / timings) and runs an end-to-end precision check against a reference
image when a model is present; it skips (exit 77) when the GGUF asset is not
downloaded.

## Models

All published GGUF models of the
[cloudViewer_downloads "sam" release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/sam)
(39 models) are listed in the model combo, grouped by family.
See [models/MODEL_CARD.md](models/MODEL_CARD.md) for the complete model catalog
with sizes and recommended defaults.

| Family | Quantizations | Text prompt |
|--------|---------------|-------------|
| `sam3-*` | f16 / q4_0 / q4_1 / q8_0 | ✅ (text detector) |
| `sam3-visual-*` | f16 / q4_0 / q4_1 / q8_0 | ❌ (visual-only) |
| `sam2.1_hiera_{tiny,small,base_plus,large}-*` | f16 / f32 / q4_0 / q4_1 / q8_0 | ❌ |
| `sam2_hiera_{tiny,base_plus,large}-*` | f16 / f32 / q4_0 / q4_1 / q8_0 | ❌ |

`sam3-f32` is intentionally not published (too large) and absent from the
catalog. Recommended default:
[`sam2.1_hiera_tiny_f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/sam/sam2.1_hiera_tiny_f16.gguf).

## Usage

### Image pages (Points / Box / Exemplar)

1. **Plugins -> SAM3 Image & Video Segmentation**
2. Pick a model in the combo (or **Browse...** a local GGUF), choose the
   device (Auto / Metal / CPU on macOS — the combo lists only the backends
   registered on this platform; Vulkan/CUDA on Linux/Windows), then **Load**
3. Open an image (file picker or drag & drop onto the canvas)
4. Prompt:
   - **Points** — left-click adds a foreground (green) point, right-click a
     background (red) point; **Segment** runs point-prompted segmentation
   - **Box (PVS)** — drag a rectangle on the canvas (cyan); **Segment** runs
     box-prompted segmentation
   - **Exemplar (PCS)** — text models only (SAM3 family): type a prompt such
     as `"cat"` and press Enter or **Segment**; optional exemplar boxes are
     supported
5. **Clear** resets the prompts; **Export masks** writes the current mask(s)
   to the DB tree as annotated images
6. Thresholds (score / NMS) and the mask overlay toggle live in the bottom
   panel; the detection list shows per-instance score / IoU / box

### Video tab (segmentation & tracking)

Qt re-implementation of the upstream `examples/main_video.cpp`:

1. Open a video file, pick a model and **Load** (a tracker is created
   automatically; SAM3 text models also accept a text prompt)
2. Choose an init mode:
   - **Text** (SAM3 family only) — auto-detect instances via the prompt
   - **Box** — drag a box on a paused frame to add an instance
   - **Points** — click positive / right-click negative points
3. **Play / Pause / Step >> / Reset** control playback; the timeline below
   the canvas shows the processed range, the playhead and one colored band
   per tracked instance (click or drag to seek)
4. Clicking an existing tracked mask refines that instance; the bottom row
   has the mask overlay toggle, playback speed, **Export frame masks** and
   the per-instance list

Video decode uses the shared `video_base` module (OpenCV / mpv), so the
video tab builds only when OpenCV capture is available
(`HAS_OPENCV_FACE_CAPTURE`).

## Outputs

- **DB tree**: annotated `ccImage` per detection (mask tint overlay +
  score/IoU/box metadata under the `SAM3/` namespace), named
  `SAM3_<source>_<device>`.

## References

- [sam3-ggml](https://github.com/Asher-1/sam3-ggml) (upstream C++ engine)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
  (SAM 2 / 2.1 / 3 models)
- [cloudViewer_downloads sam release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/sam)
