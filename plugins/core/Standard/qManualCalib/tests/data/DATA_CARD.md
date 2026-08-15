# qManualCalib 测试与示例数据

qManualCalib **不使用 GGUF 模型**；插件依赖 **车辆标定配置**（`.cfg`）与 **ROS Bag v2.0** 录制数据。

> **数据不再随源码仓库发布。** 为避免仓库携带 ~25 MB 的 bag 二进制，示例 bag 与配置打包为
> [`qcalib_test_data.zip`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qManualCalib/qcalib_test_data.zip)
> 并按需下载、缓存、解压到 `~/cloudViewer_data/extract/qcalib_test_data/`。
> 插件对话框提供 **“use test data”** 按钮一键完成下载 → 解压 → 自动填充 `configs/` 与 bag。

## 目录结构（解压后）

```
~/cloudViewer_data/extract/qcalib_test_data/
├── bags/sample_aligned.bag
└── configs/   ← Load Config 时选择此目录
```

下载 zip：`https://github.com/Asher-1/cloudViewer_downloads/releases/download/qManualCalib/qcalib_test_data.zip`

## ROS Bag 样本

| 文件 | 体积 | 时长 | 内容 | BEV 滑条 | 说明 |
|------|------|------|------|----------|------|
| `bags/sample_aligned.bag` | **24.3 MB** | ~0.6 s | SVM 7 路 + AVM 4 路 JPEG 相机；合并 LiDAR 点云 | 3 个对齐时刻（0.0 / 0.3 / 0.6 s） | GUI 试用、`test_bag_reader` |

**切片参数：** 自源 `merge.bag` 在 **15% / 50% / 85%** 处各取 1 组同步帧，重映射到输出时间线；仅保留同步图像 + 最近点云（无 ancillary 话题）。

**性能参考（`test_bag_reader`）：**

| 操作 | 典型耗时（本地 SSD） |
|------|---------------------|
| 打开 bag + 时长索引 | < 100 ms |
| `readMessageAtPercent(50%)` 单相机 JPEG | < 50 ms |
| 7 路 SVM + 4 路 AVM 并行读取 @50% | < 300 ms |
| BEV 首帧 remap（CPU，外参未变可缓存） | 200–800 ms（视分辨率与后端） |

## 标定配置文件

Load Config 时选择 `configs/`（需包含 `cameras.cfg`；可选 `lidars.cfg`、`ground.cfg`）。

| 文件 | 体积 | 用途 |
|------|------|------|
| `configs/cameras.cfg` | **8.4 KB** | 11 路相机外参 + 内参（PINHOLE / KANNALA_BRANDT） |
| `configs/lidars.cfg` | **590 B** | LiDAR 外参（`HS_ATX_SOLID`） |
| `configs/ground.cfg` | **90 B** | 地面平面约束 |
| `configs/intrinsics.cfg` | **4.3 KB** | 仅内参参考 |
| `configs/cameras.cfg_fix.cfg` | **8.5 KB** | Save Config 输出格式示例 |
| `configs/car_config.cfg` | **3.1 KB** | 车辆级配置 |
| `configs/navigation_devices.cfg` | **2.1 KB** | 导航设备占位 |

**相机模型：** `PINHOLE`、`KANNALA_BRANDT`（Equidistant）、`MEI`、`FULLPINHOLE`（见 `calib_io`）。

## 插件快速试用

| 步骤 | 操作 |
|------|------|
| 可选 | 点击 **“use test data”** 一键下载并填充（等效于步骤 1+2） |
| 1 | **Load Config** → `~/cloudViewer_data/extract/qcalib_test_data/configs/` |
| 2 | **Load Bag** → `~/cloudViewer_data/extract/qcalib_test_data/bags/sample_aligned.bag` |
| 3 | 选择 Camera / 视图模式（BEV / LiDAR Proj / Single Frame） |
| 4 | 拖动时间滑动条，6-DOF 微调外参 → **Save Config** |

菜单：**Plugins → Manual Calibration Tools → Sensor Calibration**

## 完整行车数据（需自备）

下列数据不在仓库内，需从内部数据盘或原始录制获取：

| 数据集 | 典型路径 | 体积（估） | 用途 |
|--------|----------|------------|------|
| YR-EC15S 合并 bag | `.../robotaxi_data/YR-EC15S-29_*/bags/merge.bag` | 数 GB | 重新 `mcalib_rosbag_slice` |
| YR_VF6 多 bag（HEVC） | `.../YR_VF6_1_online/bags/orig/` | 数 GB | HEVC 在线相机 + 多 bag 对齐测试 |
| Vehicle-Sample-001 | 内部数据集 `configs/` | — | 原始标定配置来源 |

重新生成 `sample_aligned.bag`：

```bash
mcalib_rosbag_slice --align-3frames \
  /path/to/source/bags/merge.bag \
  ~/cloudViewer_data/extract/qcalib_test_data/bags/sample_aligned.bag
```

详见 [`README.md`](README.md) 与插件 [`README.md`](../../README.md)。
