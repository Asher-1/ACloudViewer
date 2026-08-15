# qManualCalib 测试数据

从示例数据集 `Vehicle-Sample-001` 切分的对齐样本，供插件与 `test_bag_reader` 使用。

> **数据不再随仓库发布。** 示例 bag 与配置打包为
> [`qcalib_test_data.zip`](https://github.com/Asher-1/cloudViewer_downloads/releases/download/qManualCalib/qcalib_test_data.zip)
> 并按需下载、缓存、解压到 `~/cloudViewer_data/extract/qcalib_test_data/`，避免仓库携带 ~25 MB 的 bag 二进制。

**完整数据说明（体积、性能）见 [`DATA_CARD.md`](DATA_CARD.md)。**

## 一键获取（推荐）

打开插件对话框，点击 **“use test data”** 按钮：

1. 自动下载 `qcalib_test_data.zip` 并缓存到 `~/cloudViewer_data/download/`
2. 自动解压到 `~/cloudViewer_data/extract/qcalib_test_data/`
3. 自动加载其中的 `configs/` 与 `bags/sample_aligned.bag`

用户自定义数据仍可通过步骤 1-2（Load Config / Load Bag）正常读取，不受影响。

## 手动获取

```bash
mkdir -p ~/cloudViewer_data/extract
curl -L -o /tmp/qcalib_test_data.zip \
  https://github.com/Asher-1/cloudViewer_downloads/releases/download/qManualCalib/qcalib_test_data.zip
unzip -oq /tmp/qcalib_test_data.zip -d ~/cloudViewer_data/extract/qcalib_test_data
```

解压后目录结构：

```
qcalib_test_data/
├── bags/
│   └── sample_aligned.bag   # 多组独立对齐后合并切片（约 0.6s，24.3 MB）
└── configs/
    ├── cameras.cfg
    ├── lidars.cfg
    └── ground.cfg
```

## 数据来源

| 项 | 路径 |
|---|---|
| 原始 bag | `YR-EC15S-29_20260624_025519/bags/merge.bag`（robotaxi_data） |
| 原始配置 | `/path/to/dataset/Vehicle-Sample-001/configs/` |

原始 `merge.bag` **未被修改**，仅通过 `mcalib_rosbag_slice` 按时间窗口导出子集。

## 切片参数

- 模式：`--align-3frames` 使用 **3 组独立对齐 + 时间戳重映射合并**
  - 在源 bag 的 **15% / 50% / 85%** 附近各找一组 SVM 7 路 + AVM 4 路同步帧
  - 每组仅导出 **1 帧同步图像 + 最近点云**（`sync_frames_only`，无 ancillary 话题）
  - 3 组重映射到输出 bag 的 **0.0s / 0.3s / 0.6s** 时间线，滑条可看到 3 个不同 BEV 位置
- 时长：约 **0.6s**
- 文件大小：约 **24 MB**

**注意：** 若环视 BEV 出现黑扇区，通常是切片窗口内缺少四路 `panoramic_*` 同步帧，请用新版 `mcalib_rosbag_slice --align-3frames` 重新切分。

## 插件使用

1. **Load Config**：选择下载解压目录下的 `configs/`
2. **Load Bag**：选择下载解压目录下的 `bags/sample_aligned.bag`

## 重新生成

推荐使用项目自带工具（保留二进制 payload，避免 `rosbag filter` 损坏 `std_msgs/String`）：

**多组合并对齐切片（推荐）** — SVM/AVM/LiDAR 各组独立找同步点，重映射时间戳后写入同一 bag：

```bash
mcalib_rosbag_slice --align-3frames \
  /home/ludahai/develop/data/robotaxi_data/YR-EC15S-29_20260624_025519/bags/merge.bag \
  ~/cloudViewer_data/extract/qcalib_test_data/bags/sample_aligned.bag
```

**手动时间窗口**：

```bash
START=<bag_start_sec + offset - 0.15>
END=<bag_start_sec + offset + 0.15>
mcalib_rosbag_slice \
  /home/ludahai/develop/data/robotaxi_data/YR-EC15S-29_20260624_025519/bags/merge.bag \
  ~/cloudViewer_data/extract/qcalib_test_data/bags/sample_aligned.bag \
  "$START" "$END"
```

配置可直接从源 `configs/` 复制 `cameras.cfg`、`lidars.cfg`、`ground.cfg`。
