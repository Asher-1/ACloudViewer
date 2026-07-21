# qDA3 — Depth Anything V3 插件

在 ACloudViewer 中运行 [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) 的 **GGUF 模型**（C++ / [ggml](https://github.com/ggml-org/ggml)，源自 [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp)），用于深度估计、相机位姿与 3D 导出。

## 架构

```
GUI (qDA3 对话框) ──┐
Automatic Reconstruction (DA3DepthController) ──┼──► libAICore (depth_capi) ──► ggml
ccImage::estimateDepth* ──────────────────────────┘
```

| 组件 | 路径 |
|------|------|
| 推理库 | `core/AICore/` → `libAICore.so` |
| 插件 | `plugins/core/Standard/qDA3/` |
| 自动重建 | `libs/Reconstruction/`（`DA3DepthController`） |

## 启用与构建

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDA3=ON \
  -DBUILD_RECONSTRUCTION=ON \
  .

cmake --build build_app --target QDA3_PLUGIN ACloudViewer -j$(nproc)
```

| CMake 选项 | 说明 |
|------------|------|
| `AICore_ENABLED` | 构建 `libAICore.so`（与 qFreeSplatter 共用） |
| `PLUGIN_STANDARD_QDA3` | 本插件 |
| `BUILD_RECONSTRUCTION` | 自动重建流水线中的 DA3 稀疏/稠密模式 |
| `BUILD_CUDA_MODULE` | ggml CUDA 后端 |
| `GGML_USE_OPENCL` / `GGML_USE_VULKAN` / `GGML_USE_METAL` | 平台相关 GPU 后端（配置摘要会打印 `backends = ...`） |

产物示例：`build_app/bin/libAICore.so`、`build_app/bin/plugins/libQDA3_PLUGIN.so`。

## GUI 使用

**菜单：** Plugins → **Depth Anything V3 (DA3)** → **DA3 Depth Estimation**

1. 在 DB Tree 选中图像，或 **Browse** 选择文件。
2. 选择 **Model**（首次可 **Download** GGUF）。
3. 设置 **Device**（`Auto` / CUDA / Vulkan / CPU）、线程数、是否反投影 3D 等。
4. 点击 **Run**；深度结果以 `ccImage` 子节点加入 DB Tree。

### 模式一览

| 模式 | 输出 |
|------|------|
| Depth (single) | 深度图 → DB Tree |
| Depth + Pose | 深度 + 外参/内参 |
| Multi-view depth + pose | 多视图深度与相机 |
| 3D Reconstruct (Giant) | Giant 模型点云 |
| Export GLB / COLMAP | glTF 2.0 / COLMAP 稀疏重建目录 |
| Quantize / Model Info | GGUF 量化与元数据 |

### 推理设备（Auto）

| 平台 | Auto 优先级 |
|------|-------------|
| Linux / Windows | CUDA → OpenCL → Vulkan → CPU |
| macOS | Metal → Vulkan → CUDA → CPU |

环境变量 `DA_DEVICE` 可覆盖：`auto`、`cpu`、`cuda`、`vulkan`、`opencl[:N]`、`metal`。

### 模型缓存

| 平台 | 默认目录 |
|------|----------|
| Linux | `$HOME/cloudViewer_data/extract/da3_models` |
| Windows | `%USERPROFILE%\cloudViewer_data\extract\da3_models` |
| 覆盖 | `CLOUDVIEWER_DATA_ROOT` → `<root>/extract/da3_models` |

权重来源：[mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf)

## Automatic Reconstruction 集成

**Reconstruction → Automatic Reconstruction**：

| 设置 | 说明 |
|------|------|
| Sparse model | **DA3 (depth+pose)** — 用 DA3 多视图替代 SIFT 稀疏 |
| Stereo / dense | **DA3 depth inference** — 需 Nested AnyView + Metric 与 DA3 稀疏模式 |
| Hybrid dense (≥3 视图) | COLMAP 位姿 + DA3 metric depth 作为 PatchMatch 光度先验，可选跳过几何 refine（UI / `DA3_SKIP_GEOMETRIC_REFINE`） |

## C API（简要）

头文件：`core/AICore/include/aicore/depth_capi.h`

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

## 延伸阅读

- 完整插件 README（测试、parity、脚本）：[`plugins/core/Standard/qDA3/README.md`](../../../plugins/core/Standard/qDA3/README.md)
- [Depth Anything 3](https://github.com/DepthAnything/Depth-Anything-V3) · [depth-anything.cpp](https://github.com/mudler/depth-anything.cpp)
