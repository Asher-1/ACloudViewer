# qManualCalib — 手动标定与环视调整

ACloudViewer 插件，用于 **自动驾驶传感器外参手动微调** 与 **AVM（环视）视图参数调整**。读取 Protobuf 文本标定配置与自研 ROS Bag v2.0 解析器，**无需 GGUF / AICore**。

## 功能模块

| 模块 | 菜单 | 用途 |
|------|------|------|
| **Manual Sensor Calibration** | Plugins → Manual Calibration Tools → **Sensor Calibration** | 相机 / LiDAR 6-DOF 外参、BEV 拼接、点云投影 |
| **AVM View Adjustment** | Plugins → Manual Calibration Tools → **AVM Adjustment** | 环视 remap 参数（14 项）实时调整 |

## 启用与构建

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DBUILD_OPENCV=ON \
  -DPLUGIN_STANDARD_QMANUAL_CALIB=ON \
  .

cmake --build build_app --target QMANUAL_CALIB_PLUGIN ACloudViewer -j$(nproc)
```

| CMake 选项 | 说明 |
|------------|------|
| `PLUGIN_STANDARD_QMANUAL_CALIB` | 本插件（默认 OFF） |
| `MCALIB_WITH_FFMPEG_SUPPORT` | H.264/HEVC 相机 topic 解码（默认 ON，需系统 FFmpeg） |
| `MCALIB_BEV_CUDA` / `MCALIB_BEV_OPENCL` | BEV remap GPU 后端 |
| `MCALIB_BUILD_TESTS` | `test_bag_reader` |
| `MCALIB_BUILD_TOOLS` | `mcalib_*` CLI 工具 |

产物：`build_app/bin/plugins/libQMANUAL_CALIB_PLUGIN.so`

启用插件时会额外编译 OpenCV `calib3d`、`objdetect`（ArUco）。

## 示例数据

无需自备行车 bag 即可试用：

| 项 | 路径 |
|----|------|
| 配置 | `plugins/core/Standard/qManualCalib/tests/data/configs/` |
| 对齐样本 bag | `plugins/core/Standard/qManualCalib/tests/data/bags/sample_aligned.bag` |

完整体积与性能说明见 [`tests/data/DATA_CARD.md`](../../../plugins/core/Standard/qManualCalib/tests/data/DATA_CARD.md)（数据随插件源码集成）。

### Sensor Calibration 快速流程

1. **Load Config** → 选择 `configs/` 目录  
2. **Load Bag** → 选择 `sample_aligned.bag` 或 bag **目录**（支持多 bag 自动发现）  
3. 选择传感器、标定模式（single / all / avm / svm）、视图（BEV / LiDAR Proj / Single Frame）  
4. 拖动时间滑动条，用 Roll/Pitch/Yaw/X/Y/Z 微调  
5. **Save Config** → `cameras_fix.cfg` / `lidars_fix.cfg`，或 **Export Image / PCD / BEV**

### BEV Remap 后端

UI 可选 **Auto / CUDA / OpenCL / CPU**；GPU 失败自动回退 CPU。外参不变时 remap 表缓存复用。

### ROS Bag 布局

| 布局 | 说明 |
|------|------|
| **SingleFile** | 单个 `merge.bag` |
| **FlatTopicGroup** | `bags/orig/` 下 Heavy/Light/Medium 多 bag，`openMulti` 时间对齐 |
| **NestedTopicGroup** | 嵌套 `raw_bags/` 目录 |

在线 **HEVC/H.264** 相机需 FFmpeg；解码状态在 bag 读取器内缓存。

## 可选 CLI

`-DMCALIB_BUILD_TOOLS=ON` 后可用：

| 工具 | 用途 |
|------|------|
| `mcalib_rosbag_slice` | 时间切片 / `--align-3frames` 多组对齐合并 |
| `mcalib_export_bev` | 无头批量 BEV 导出 |
| `mcalib_rosbag2image` / `mcalib_rosbag2pcd` | bag → JPG / PCD |
| `mcalib_extrinsic_compare` | 外参 diff |

详见 [`tools/README.md`](../../../plugins/core/Standard/qManualCalib/tools/README.md)。

## 测试

```bash
cmake -B build_app \
  -DPLUGIN_STANDARD_QMANUAL_CALIB=ON \
  -DMCALIB_BUILD_TESTS=ON \
  -DBUILD_OPENCV=ON \
  .

cmake --build build_app --target test_bag_reader -j$(nproc)
./build_app/bin/plugins/test_bag_reader
```

测试数据目录由 `MCALIB_TEST_DATA_DIR` 注入（默认 `tests/data`）。

## 坐标系说明

Manual Calibration 使用 **车辆 / 传感器配置坐标**（BEV、投影视图），与 COLMAP 重建或 qFreeSplatter PLY 的 OpenGL 坐标无直接对齐关系。

## 延伸阅读

- 开发者 README：[`plugins/core/Standard/qManualCalib/README.md`](../../../plugins/core/Standard/qManualCalib/README.md)
- 示例数据卡片：[`tests/data/DATA_CARD.md`](../../../plugins/core/Standard/qManualCalib/tests/data/DATA_CARD.md)
