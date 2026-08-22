# qRMBG — RMBG-2.0 Background Removal Plugin

Run **RMBG-2.0 (BiRefNet-Swin-L) GGUF** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for high-quality image background removal with transparent output.

## Architecture

```
GUI (RMBG dialog) ──► libAICore (rmbg_capi) ──► GGML BiRefNet
                     ├── remove_background_rgb → RGBA PNG at original resolution
                     └── alpha_mat_rgb          → raw 8-bit alpha matte (future plugins)
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| GGML RMBG-2.0 engine | `core/AICore/src/tasks/rmbg/` (port of [RMBG-2.0-GGML](https://github.com/Asher-1/RMBG-2.0-GGML)) |
| Plugin | `plugins/core/Standard/qRMBG/` |
| ggml patch | `3rdparty/ggml/patches/rmbg_merged/` (CUDA kernels + Vulkan shaders) |

`aicore_rmbg_*` is the **foundational background-removal module**: the raw
alpha matte API lets future plugins threshold / feather / re-composite at
their own resolution.

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QRMBG=ON \
  .

cmake --build build_app --target QRMBG_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` (shared with qDA3, qDeepLSD, qLightGlue, qFreeSplatter, qRFDetr) |
| `PLUGIN_STANDARD_QRMBG` | This plugin |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQRMBG_PLUGIN.so`.

## GUI usage

**Menu:** Plugins → **RMBG Remove Background**

### Image tab

1. Select the model (default `rmbg_f16.gguf`; f32 / q8 variants are also offered in the combo).
2. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads** (0 = auto).
3. Pick an input image from disk or the DB tree (collapsible DB list, or select in the main DB tree).
4. Click **Run** — the model downloads from cloudViewer_downloads on first use.

The transparent RGBA result is added to the DB tree as `RMBG_<source>_<device>`
(metadata: alpha mean, foreground ratio, runtime, backend, device, model).
Optionally save the PNG to a directory of your choice. The preview composites
the result over a checkerboard so removed background is clearly visible.

### Live (camera / video) tab

1. Start the camera or open a video file (reuses `video_base` playback).
2. Playback is inference-paced: a decoded frame is displayed only after its background-removal result has been drawn, so the preview always matches the current frame over the checkerboard.
3. **Capture** stores the current RGBA snapshot into the DB tree.

## Models

Official weights: [cloudViewer_downloads trellis2-ggml release](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/trellis2-ggml) — unified encoder + decoder GGUF in three quantizations.

| Filename | Size (approx.) | Notes |
|----------|----------------|-------|
| `rmbg_f16.gguf` | ~420 MB | BiRefNet-Swin-L, 1024×1024 input (**default**) |
| `rmbg_f32.gguf` | ~840 MB | float32 reference |
| `rmbg_q8.gguf`  | ~245 MB | 8-bit quant, smallest download |

Model cache directory: `rmbg_models/`.

See [MODEL_CARD.md](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qRMBG/models/MODEL_CARD.md) for download links and licensing.

## Backends

- **CUDA**: custom `rmbg-deform-im2col` / `rmbg-conv2d` / `rmbg-swin` kernels
- **Vulkan**: 7 custom shaders
- **CPU / Metal**: automatic vanilla ggml operator fallback

Device selection follows AICore's `Auto` order (CUDA → Vulkan → CPU on
Linux/Windows, Metal → CPU on macOS) and can be overridden in the dialog.
