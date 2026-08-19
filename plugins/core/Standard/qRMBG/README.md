# qRMBG

RMBG-2.0 (BiRefNet-Swin-L) background removal for ACloudViewer — **native C++ GGML**.

**User guide:** [docs/guides/plugins/qRMBG.md](../../../docs/guides/plugins/qRMBG.md)

```
Image/Video → AICore RMBG-2.0 GGML → RGBA composite → ccImage (transparent) → DB tree
```

## Build

```bash
cmake -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QRMBG=ON \
  ..
make -j4 QRMBG_PLUGIN
```

RMBG-2.0 GGML sources live in `core/AICore/src/tasks/rmbg/` (in-tree port of
[RMBG-2.0-GGML](https://github.com/Asher-1/RMBG-2.0-GGML)). The custom CUDA /
Vulkan operators are applied as ggml patches (`3rdparty/ggml/patches/rmbg_merged/`);
CPU and Metal fall back to vanilla ggml operators automatically.

### Backend profiles and timing

CUDA and Vulkan use the upstream optimized graph and kernels by default. Set
`RMBG_STRICT_MATH=1` for the strict CUDA path, or select a Vulkan profile with
`RMBG_VULKAN_MODE=optimized|strict|unsafe-fast`. `unsafe-fast` enables all
available Vulkan fast-math paths and should only be used after validating
output parity on the target GPU.

The reported `Runtime (ms)` is only `graph->forward()`, matching the upstream
benchmark. Separate preprocess, postprocess and total values are stored in the
result metadata.

### Unit tests (pure helpers)

Alpha statistics, checkerboard preview and catalog mirror (no GGUF model
required):

```bash
cmake -DBUILD_GUI=ON -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QRMBG=ON \
  -DBUILD_UNIT_TESTS=ON ..
cmake --build build_app --target test_qrmbg_helpers -j4
ctest -R test_qrmbg_helpers --output-on-failure
```

The real-model regression test uses two warmups and seven measured runs. A
hardware-specific median ceiling can be enabled in CI or locally:

```bash
AICORE_TEST_RMBG_GGUF=/path/to/rmbg_f16.gguf \
AICORE_TEST_RMBG_MAX_MEDIAN_MS=660 \
./build_app/bin/aicore_tests/test_rmbg_capi_performance cuda
```

## Models

See [models/MODEL_CARD.md](models/MODEL_CARD.md). The model combo lists three
quantizations of the unified encoder+decoder GGUF:

- [`rmbg_f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/trellis2-ggml/rmbg_f16.gguf)
  (~420 MB) — **default**
- [`rmbg_f32.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/trellis2-ggml/rmbg_f32.gguf)
  (~840 MB) — float32 reference
- [`rmbg_q8.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/trellis2-ggml/rmbg_q8.gguf)
  (~245 MB) — 8-bit quant, smallest download

(`rmbg_q8_0.gguf` / `rmbg_q4_K.gguf` are **not** published upstream; they are
not offered in the model combo.)

## Usage

### Image tab

1. **Plugins → RMBG Remove Background**
2. Select the model (auto-downloads on first Run if missing)
3. Pick an image from disk or DB tree → **Run**
4. The transparent RGBA result is added to the DB tree and/or saved as PNG

The image preview shows the result composited over a checkerboard so removed
background is clearly visible.

### Live (camera / video) tab

Play a video file or use the camera. Playback is inference-paced: the next frame
is decoded only after the current background-removal result has been displayed.
Snapshot the current result into the DB tree with the capture button.

## Outputs

- **DB tree**: `ccImage` named `RMBG_<source>_<device>`, metadata includes
  alpha mean / foreground ratio, runtime, backend, device and model filename.
- **PNG**: optional save of the RGBA composite to a user-selected directory.

## References

- [RMBG-2.0](https://github.com/Bria-AI/RMBG-2.0) (upstream PyTorch)
- [RMBG-2.0-GGML](https://github.com/Asher-1/RMBG-2.0-GGML) (upstream ggml)