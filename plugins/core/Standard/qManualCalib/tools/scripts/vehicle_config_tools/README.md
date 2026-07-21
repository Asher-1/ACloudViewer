# 车辆配置文件更新工具

从 Excel 传感器参数表更新车辆配置文件。

## 快速开始

### GUI 版本（推荐）
```bash
conda activate env_calib_py310
unset LD_LIBRARY_PATH  # 避免与系统 Qt 冲突
python generate_vehicle_config_gui.py
```

### 命令行版本
```bash
conda activate env_calib_py310
python generate_vehicle_config.py \
  --excel "传感器参数表.xlsx" \
  --raw-config "raw_config" \
  --output-config "output_config" \
  --main-radar-type normal_radar
```

## 车厂配置

不同车厂使用不同的 `raw_config` 模板目录，下表为示例命名（请按实际交付包调整）：

| 车厂 | raw_config 目录 | Radar 数量 |
|----------|----------------|------------|
| OEM-A | raw_config_oem_a | 3 |
| OEM-B | raw_config_oem_b | 3 |
| OEM-C | raw_config_oem_c | 3 |
| OEM-D | raw_config_oem_d | 5 |
| OEM-E | raw_config_oem_e | 5 |

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--excel` | ✅ | Excel 传感器参数表路径 |
| `--raw-config` | ✅ | 原始配置目录 |
| `--output-config` | ✅ | 输出配置目录 |
| `--main-radar-type` | ✅ | 主雷达类型: `normal_radar` 或 `4d_radar` |

## 更新的配置文件

| 文件 | 更新内容 |
|------|---------|
| cameras.cfg | 11 个相机外参 |
| lidars.cfg | 多 LiDAR 外参（从 lidar_node.cfg 自动提取）+ vehicle_to_sensing |
| ultrasonics.cfg | 12 个超声波传感器外参（自动识别厂商前缀） |
| navigation_devices.cfg | GNSS 天线位置 (mm→m) |
| radars.cfg | 根据雷达配置更新位置（支持 3/5 颗 Radar） |
| car_config.cfg | 直接复制 |

## LiDAR 配置规则

从 `lidar_node.cfg` 自动提取所有 LiDAR 的 `frame_id` 和 `type`，支持任意数量的 LiDAR。

**Excel 映射规则（基于 frame_id 后缀 `_20X`）：**

| frame_id 后缀 | 对应 Excel 名称 |
|--------------|----------------|
| `_201` | NLiDAR 4 |
| `_202` | LiDAR 1 |
| `_203` | NLiDAR 2 |
| `_204` | NLiDAR 1 |
| `_205` | NLiDAR 3 |
| `_206` | FLiDAR 1 |
| `_207` | FLiDAR 3 |
| `_208` | FLiDAR 2 |

**示例：**

- `lidar_qt_128_201` → Excel 的 `NLiDAR 4`
- `atx_202` → Excel 的 `LiDAR 1`
- `lidar_64_207` → Excel 的 `FLiDAR 3`

**注意：** 如果 `lidar_node.cfg` 中的 LiDAR model 在 proto 中不存在，程序会报错退出，需要更新 proto 定义。

## 雷达配置组合

| Radar 数量 | 主雷达类型 | 适用变体 |
|-----------|-----------|----------|
| 3 | normal_radar | OEM-A / OEM-B / OEM-C |
| 3 | 4d_radar | 4D 主雷达配置 |
| 5 | normal_radar | OEM-D / OEM-E |

## USS 厂商自动识别

根据 Excel 中的传感器型号自动识别 USS 厂商，并更新 `frame_id` 前缀（Excel 厂商列需与代码键一致）：

- `VENDOR_A` → `vendor_a_`
- `VENDOR_B` → `vendor_b_`
- `VENDOR_C` → `vendor_c_`

## 文件说明

| 文件 | 说明 |
|------|------|
| generate_vehicle_config.py | 核心逻辑（命令行版本） |
| generate_vehicle_config_gui.py | GUI 版本（PyQt5，含 3D 可视化） |
| generate_vehicle_config_node.cpp | C++ 版本，生成 sensing 坐标系配置 |

## GUI 功能

- 车厂选择（自动填充 raw_config 路径）
- Excel 文件拖拽支持
- 自动生成输出目录
- 3D 传感器可视化
- 一键生成 sensing 坐标系配置

## 依赖

- pandas, openpyxl — Excel 读取
- protobuf>=3.19,<4 — 配置文件解析
- PyQt5, pyqtgraph, PyOpenGL — GUI 界面和 3D 可视化
