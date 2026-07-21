# qManualCalib 测试数据

从示例数据集 `Vehicle-Sample-001` 切分的对齐样本，供插件与 `test_bag_reader` 使用。

**完整数据说明（体积、性能）见 [`DATA_CARD.md`](DATA_CARD.md)。**

## 目录结构

```
data/
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

1. **Load Config**：选择 `tests/data/configs/`
2. **Load Bag**：选择 `tests/data/bags/sample_aligned.bag`

## 重新生成

推荐使用项目自带工具（保留二进制 payload，避免 `rosbag filter` 损坏 `std_msgs/String`）：

**多组合并对齐切片（推荐）** — SVM/AVM/LiDAR 各组独立找同步点，重映射时间戳后写入同一 bag：

```bash
mcalib_rosbag_slice --align-3frames \
  /home/ludahai/develop/data/robotaxi_data/YR-EC15S-29_20260624_025519/bags/merge.bag \
  tests/data/bags/sample_aligned.bag
```

**手动时间窗口**：

```bash
START=<bag_start_sec + offset - 0.15>
END=<bag_start_sec + offset + 0.15>
mcalib_rosbag_slice \
  /home/ludahai/develop/data/robotaxi_data/YR-EC15S-29_20260624_025519/bags/merge.bag \
  tests/data/bags/sample_aligned.bag \
  "$START" "$END"
```

配置可直接从源 `configs/` 复制 `cameras.cfg`、`lidars.cfg`、`ground.cfg`。
