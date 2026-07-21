# AI 推理插件（AICore）

ACloudViewer 通过统一推理库 **`libAICore.so`**（ggml）提供两类 AI 插件，均**无需 Python/PyTorch 运行时**：

| 插件 | 文档 | CMake 选项 | 功能摘要 |
|------|------|------------|----------|
| **qDA3** | [qDA3 使用指南](qDA3.md) | `PLUGIN_STANDARD_QDA3` | Depth Anything V3：单目/多视图深度、相机位姿、COLMAP/GLB 导出、自动重建集成 |
| **qFreeSplatter** | [qFreeSplatter 使用指南](qFreeSplatter.md) | `PLUGIN_STANDARD_QFREESPLATTER` | FreeSplatter：无标定照片 → 3D Gaussian Splatting、SIBR 兼容 PLY、可选 qSIBR 预览 |

## 前置条件

- `-DAICore_ENABLED=ON`（构建 `core/AICore` → `libAICore.so`）
- GUI：`-DBUILD_GUI=ON`
- **qDA3 + 自动重建**：`-DBUILD_RECONSTRUCTION=ON`
- **FreeSplatter 一键 Visualize**：`-DPLUGIN_STANDARD_QSIBR=ON`（Linux/Windows；macOS CI 默认关闭 qSIBR）
- **GPU 加速（推荐）**：`-DBUILD_CUDA_MODULE=ON`；可选 ggml OpenCL/Vulkan/Metal（见各插件文档）

## 典型构建

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDA3=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  -DBUILD_RECONSTRUCTION=ON \
  .

cmake --build build_app --target ACloudViewer QDA3_PLUGIN QFREESPLATTER_PLUGIN -j$(nproc)
```

## 标定插件（qManualCalib）

非 AICore 插件，示例数据随源码 `tests/data/` 集成，无需额外下载。

| 插件 | 文档 | CMake 选项 |
|------|------|------------|
| **qManualCalib** | [qManualCalib 使用指南](qManualCalib.md) | `PLUGIN_STANDARD_QMANUAL_CALIB` |

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DBUILD_OPENCV=ON \
  -DPLUGIN_STANDARD_QMANUAL_CALIB=ON \
  .

cmake --build build_app --target QMANUAL_CALIB_PLUGIN ACloudViewer -j$(nproc)
```

## 更多资料

- 插件目录完整 README（开发者细节、测试、C API）：[`plugins/core/Standard/qDA3/README.md`](../../../plugins/core/Standard/qDA3/README.md)、[`plugins/core/Standard/qFreeSplatter/README.md`](../../../plugins/core/Standard/qFreeSplatter/README.md)
- qManualCalib 开发者 README：[`plugins/core/Standard/qManualCalib/README.md`](../../../plugins/core/Standard/qManualCalib/README.md)
- 插件总索引：[`plugins/README.md`](../../../plugins/README.md)
- Sphinx 构建时会将上述 README 同步到 `docs/source/plugins/`（见 `docs/source/conf.py`）
