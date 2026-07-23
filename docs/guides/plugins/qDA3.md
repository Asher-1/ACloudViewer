# qDA3 — Depth Anything V3 Plugin

Run [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) **GGUF models** in ACloudViewer (C++ / [ggml](https://github.com/ggml-org/ggml), derived from [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp)) for depth estimation, camera pose, and 3D export.

## Architecture

```
GUI (qDA3 dialog) ──┐
Automatic Reconstruction (DA3DepthController) ──┼──► libAICore (depth_capi) ──► ggml
ccImage::estimateDepth* ──────────────────────────┘
```

| Component | Path |
|-----------|------|
| Inference library | `core/AICore/` → `libAICore.so` |
| Plugin | `plugins/core/Standard/qDA3/` |
| Automatic reconstruction | `libs/Reconstruction/` (`DA3DepthController`) |

## Enable and build

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDA3=ON \
  -DBUILD_RECONSTRUCTION=ON \
  .

cmake --build build_app --target QDA3_PLUGIN ACloudViewer -j$(nproc)
```

| CMake option | Description |
|--------------|-------------|
| `AICore_ENABLED` | Build `libAICore.so` (shared with qFreeSplatter) |
| `PLUGIN_STANDARD_QDA3` | This plugin |
| `BUILD_RECONSTRUCTION` | DA3 sparse/dense modes in the automatic reconstruction pipeline |
| `GGML_USE_VULKAN` / `GGML_USE_METAL` | Linux/Windows: Vulkan ON; macOS: Metal ON |
| `GGML_USE_SYCL` / `GGML_USE_CUDA` | Optional Intel/NVIDIA developer backends |

Example outputs: `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQDA3_PLUGIN.so`.

## GUI usage

**Menu:** Plugins → **Depth Anything V3 (DA3)** → **DA3 Depth Estimation**

1. Select image(s) in the DB tree, or use **Browse** to pick files.
2. Choose a **Model** (use **Download** on first run for GGUF weights).
3. Set **Device** (`Auto` / Metal / SYCL / Vulkan / CUDA / CPU, available entries only), thread count, unproject-to-3D options, etc.
4. Click **Run**; depth results appear as `ccImage` child nodes in the DB tree.

### Modes

| Mode | Output |
|------|--------|
| Depth (single) | Depth map → DB tree |
| Depth + Pose | Depth + extrinsics / intrinsics |
| Multi-view depth + pose | Multi-view depth and cameras |
| 3D Reconstruct (Giant) | Giant-model point cloud |
| Export GLB / COLMAP | glTF 2.0 / COLMAP sparse reconstruction folder |
| Quantize / Model Info | GGUF quantization and metadata |

### Inference device (Auto)

| Platform | Auto priority |
|----------|---------------|
| macOS | Metal → CPU |
| Linux / Windows | Vulkan → CPU |

Override with `DA_DEVICE`: `auto`, `cpu`, `sycl[:N]`, `vulkan[:N]`, `cuda[:N]`, `metal`. SYCL/CUDA are explicit developer devices and are not selected by Auto.

### Model cache

| Platform | Default directory |
|----------|-------------------|
| Linux | `$HOME/cloudViewer_data/extract/da3_models` |
| Windows | `%USERPROFILE%\cloudViewer_data\extract\da3_models` |
| Override | `CLOUDVIEWER_DATA_ROOT` → `<root>/extract/da3_models` |

Weights: [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf)

## Automatic Reconstruction integration

**Reconstruction → Automatic Reconstruction** (`libs/Reconstruction` + `app/reconstruction` share `colmap::AutomaticReconstructionController`):

| Setting | Description |
|---------|-------------|
| Sparse model | COLMAP SfM or **DA3 (depth+pose)**; with ≥3 images, hybrid mode (COLMAP poses + DA3 depth) is available |
| Stereo / dense | **DA3 depth inference** (Nested AnyView/Metric) or COLMAP PatchMatch (**CUDA only**) |
| No CUDA + AICore | Defaults to DA3 sparse/stereo; full dense path: undistort → DA3 depth → voxel fusion → mesh → texturing |
| Hybrid dense (≥3 views, CUDA) | COLMAP poses + DA3 priors + optional PatchMatch refine + StereoFusion |

If the UI still selects COLMAP PatchMatch but the build has no CUDA, runtime falls back to DA3 stereo via `EffectiveStereoPipelineMode`.

## C API (brief)

Header: `core/AICore/include/aicore/depth_capi.h`

```c
#include "aicore/depth_capi.h"

aicore_depth_ctx* ctx = aicore_depth_load("model.gguf", 8);
int h, w, is_metric;
float *depth, *conf, ext[12], intr[9];
aicore_depth_depth_dense(ctx, "photo.jpg", &h, &w, &depth, &conf, NULL,
                         ext, intr, &is_metric);
aicore_depth_free_floats(depth);
aicore_depth_free(ctx);
```

## Further reading

- Full plugin README (tests, parity, scripts): [`plugins/core/Standard/qDA3/README.md`](../../../plugins/core/Standard/qDA3/README.md)
- [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) · [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp)
