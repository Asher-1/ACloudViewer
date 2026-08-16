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

### Unit tests (pure helpers)

Alpha statistics, checkerboard preview and catalog mirror (no GGUF model
required):

```bash
cmake -DBUILD_GUI=ON -DAICore_ENABLED=ON -DPLUGIN_STANDARD_QRMBG=ON \
  -DBUILD_UNIT_TESTS=ON ..
cmake --build build_app --target test_qrmbg_helpers -j4
ctest -R test_qrmbg_helpers --output-on-failure
```

## Models

See [models/MODEL_CARD.md](models/MODEL_CARD.md). Default download:

[`rmbg_f16.gguf`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/trellis2-ggml/rmbg_f16.gguf)
(unified encoder + decoder, F16)

## Usage

### Image tab

1. **Plugins → RMBG Remove Background**
2. Select the model (auto-downloads on first Run if missing)
3. Pick an image from disk or DB tree → **Run**
4. The transparent RGBA result is added to the DB tree and/or saved as PNG

The image preview shows the result composited over a checkerboard so removed
background is clearly visible.

### Live (camera / video) tab

Play a video file or use the camera; inference is throttled (every 5th video
frame) and the preview shows the background-removed result in real time.
Snapshot the current result into the DB tree with the capture button.

## Outputs

- **DB tree**: `ccImage` named `RMBG_<source>_<device>`, metadata includes
  alpha mean / foreground ratio, runtime, backend, device and model filename.
- **PNG**: optional save of the RGBA composite to a user-selected directory.
