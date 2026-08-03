# qDeepLSD — DeepLSD Line Extraction Plugin

Run [DeepLSD](https://github.com/cvg/DeepLSD) **GGUF models** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml)) for wireframe line-segment extraction from images.

## Architecture

```
GUI (DeepLSD dialog) ──► libAICore (deeplsd_capi) ──► GGML CNN ──► df + angle fields
                                                              └──► AFM + LSD post-process ──► line segments
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| GGML DeepLSD CNN | `core/AICore/src/deeplsd/` |
| LSD post-process | `core/AICore/src/deeplsd/deeplsd_line_detect.cpp` (pytlsd / LSD) |
| Plugin | `plugins/core/Standard/qDeepLSD/` |

The GGML path outputs **distance + angle fields**; line segments are extracted in C++ with the same AFM-guided LSD pipeline used in DeepLSD-GGML validation scripts.

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDEEPLSD=ON \
  .

cmake --build build_app --target QDEEPLSD_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` (shared with qDA3, qLightGlue, qFreeSplatter) |
| `PLUGIN_STANDARD_QDEEPLSD` | This plugin |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQDEEPLSD_PLUGIN.so`.

## GUI usage

**Menu:** Plugins → **DeepLSD Wireframe** → **DeepLSD Line Extraction**

1. Choose a **GGUF model** (wireframe for indoor / CAD-like scenes, MegaDepth for outdoor).
2. Set **Device** (`Auto` / CUDA / Vulkan / CPU) and **Threads**.
3. Pick an **input image** from disk or the DB tree (click the thumbnail to enlarge).
4. Choose DB export options (see below) and click **Run**.

Models download on first run if missing (progress shows downloaded / total size).

### Model variants

| Variant | Training data | Best for |
|---------|---------------|----------|
| **Wireframe** (`deeplsd_wireframe-*`) | Synthetic wireframe + ScanNet | Indoor, man-made geometry, CAD-like edges |
| **MegaDepth** (`deeplsd_md-*`) | Outdoor phototourism | Natural scenes, facades, street photography |

Recommended default: **Wireframe F16**. Q8_0 variants are smaller and suitable for quick tests.

See [MODEL_CARD.md](../../../plugins/core/Standard/qDeepLSD/models/MODEL_CARD.md) for download links.

### DB export options

| Checkbox | Default | Output |
|----------|---------|--------|
| Add line-segment visualization | On | `ccImage` with green line overlay on grayscale source |
| Add distance-field heatmap overlay | Off | `ccImage` with red df heatmap (research / debug) |
| Export detected segments as ccPolyline | Off | Group of 2D `ccPolyline` entities (green, one segment each) |

The **line visualization** is the normal “detected segments” view. The distance-field overlay shows the raw neural **df field** (hence the red heatmap look), not LSD lines.

### Inference device (Auto)

| Platform | Auto priority |
|----------|---------------|
| macOS | Metal → CPU |
| Linux / Windows | CUDA → Vulkan → CPU (when CUDA backend is built) |

### Model cache

| Platform | Default directory |
|----------|-------------------|
| Linux | `$HOME/cloudViewer_data/extract/deeplsd_models` |
| Windows | `%USERPROFILE%\cloudViewer_data\extract\deeplsd_models` |
| Override | `CLOUDVIEWER_DATA_ROOT` → `<root>/extract/deeplsd_models` |

Default download: [deeplsd_wireframe-f16.gguf](https://github.com/Asher-1/cloudViewer_downloads/releases/download/DeepLSD/deeplsd_wireframe-f16.gguf)

## C API (brief)

Header: `core/AICore/include/aicore/deeplsd_capi.h`

```c
#include "aicore/deeplsd_capi.h"

aicore_deeplsd_options* opts = aicore_deeplsd_options_new();
aicore_deeplsd_options_set_device(opts, "auto");
aicore_deeplsd_ctx* ctx = aicore_deeplsd_load_opts("deeplsd_wireframe-f16.gguf", opts);

float* df = NULL;
float* ang = NULL;
aicore_deeplsd_segment* segs = NULL;
int32_t seg_count = 0;
int32_t w = 0, h = 0;
aicore_deeplsd_extract_segments(ctx, gray, width, height, stride,
                                &segs, &seg_count, &df, &ang, &w, &h);

free(df);
free(ang);
free(segs);
aicore_deeplsd_free(ctx);
aicore_deeplsd_options_free(opts);
```

## Further reading

- Developer README (build targets, tests): [`plugins/core/Standard/qDeepLSD/README.md`](../../../plugins/core/Standard/qDeepLSD/README.md)
- GGML parity notes: DeepLSD-GGML `cpp/BENCHMARK.md`
- [DeepLSD (CVG)](https://github.com/cvg/DeepLSD)
