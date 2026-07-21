# qFreeSplatter — FreeSplatter 3D Gaussian Splatting

将普通照片转为 **3D Gaussian Splatting** 点云 — **无需相机位姿、无需 Python**；与 qDA3 共用 `libAICore.so`（ggml 推理 [FreeSplatter](https://github.com/TencentARC/FreeSplatter)）。

## 工作流

```
输入图像 (2+ 张) → FreeSplatterDialog → FreeSplatterWorker → libAICore (gaussian_capi)
    → SIBR 兼容 PLY → DB Tree → [可选] qSIBR Gaussian Viewer
```

## 启用与构建

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  .

cmake --build build_app --target QFREESPLATTER_PLUGIN ACloudViewer -j$(nproc)
```

| CMake 选项 | 说明 |
|------------|------|
| `AICore_ENABLED` | 构建含 FreeSplatter 的 `libAICore.so` |
| `PLUGIN_STANDARD_QFREESPLATTER` | 本插件 |
| `PLUGIN_STANDARD_QSIBR` | 可选；开启后显示 **Visualize (SIBR)** 按钮（运行时调用，无静态链接） |
| `PLUGIN_STANDARD_QFREESPLATTER_TOOLS` | 可选 CLI `free_splatter-cli` |
| `AICore_BUILD_TESTS` | `core/AICore/tests/gaussian/` 单元测试 |

## GUI 使用

**菜单：** Plugins → **FreeSplatter 3D Reconstruction**

| 步骤 | 操作 |
|------|------|
| 1 | 选择 **Model** 类型：Scene（2 视图）或 Object（3+ 视图） |
| 2 | 选择 **GGUF 模型**（F16/F32/Q8_0；首次可自动下载） |
| 3 | **Add Images**：文件、文件夹或 DB Tree 多选 |
| 4 | **Device**：`Auto` / CUDA / Vulkan / CPU |
| 5 | **Run** → 导出 PLY，可选 **Add to DB** |
| 6 | **Visualize**（需 `PLUGIN_STANDARD_QSIBR=ON`）→ 启动 qSIBR Gaussian Viewer |

### 输入约束

| 模型 | 最少图像 | 用途 |
|------|----------|------|
| Scene | **2** 张 | 室内/室外场景 |
| Object | **3+** 张 | 单物体 |

可选：**Estimate poses**（PnP）、**Opacity threshold**、Basic/Full PLY 字段。

### 推理设备（Auto）

与 qDA3 相同：Linux/Windows 为 CUDA → OpenCL → Vulkan → CPU；macOS 为 Metal → Vulkan → CUDA → CPU。

### 模型与缓存

自动下载源：[cloudViewer_downloads/3dgs](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs)

| 类型 | 推荐文件 | 约大小 |
|------|----------|--------|
| Scene F16 | `freesplatter-scene-f16.gguf` | ~400 MB |
| Object F16 | `freesplatter-object-f16.gguf` | ~400 MB |

缓存目录：`~/cloudViewer_data/extract/freesplatter_models`（可用 `CLOUDVIEWER_DATA_ROOT` 覆盖）。

## 输出

- **PLY**：SIBR / 3D Gaussian Splatting 查看器兼容（OpenGL 坐标系，含 SH、opacity、scale、rotation）
- **DB Tree**：点云实体，命名带 `FS_` 前缀与模型类型标签（见 `ecvPluginDbNaming`）

## 与 qSIBR 联动

1. 运行 FreeSplatter 得到 PLY  
2. 点击 **Visualize**，或手动：Plugins → SIBR → **3D Gaussian Splatting Viewer**  
3. macOS 上 qSIBR 可能因 OpenGL 限制未启用；仍可导出 PLY 在外部查看

## 测试（可选）

```bash
cmake -B build -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON ...
cmake --build build --target test_loader test_parity
ctest -LE model   # 无 GGUF 的快速测试
```

## 延伸阅读

- 完整插件 README：[`plugins/core/Standard/qFreeSplatter/README.md`](../../../plugins/core/Standard/qFreeSplatter/README.md)
- [FreeSplatter](https://github.com/TencentARC/FreeSplatter) · [free-splatter.cpp](https://github.com/LocalAI-io/free-splatter.cpp)
