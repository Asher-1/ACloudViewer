#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆配置更新脚本
从Excel文件中提取传感器外参数据，更新所有配置文件

根据PDF说明更新以下文件：
- cameras.cfg: 11个相机外参
- lidars.cfg: 激光雷达外参
- ultrasonics.cfg: 超声波传感器外参
- navigation_devices.cfg: GNSS天线位置
- radars.cfg: 雷达位置（只更新xyz，角度不更新）
- car_config.cfg: 不操作（复制原文件）

转换规则：
- position: X, Y, Z (单位：mm，直接使用)
- orientation: ψ → qz, θ → qy, φ → qx, qw = 0
- radars: 只更新position，orientation保持默认值(qw=1)
"""

import sys

# 检查Python版本
if sys.version_info < (3, 6):
    print("错误: 需要Python 3.6或更高版本")
    print(f"当前版本: {sys.version}")
    sys.exit(1)

import pandas as pd
import re
import shutil
import math
import os
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# 添加 proto 路径
sys.path.insert(0, '/path_to/common-protocol/include/proto/')

from google.protobuf import text_format
from drivers.camera.config_pb2 import Configs as CameraConfigs
from drivers.lidar.config_pb2 import LidarConfig
from drivers.lidar.lidar_type_pb2 import Model as LidarModel
from drivers.ultrasonic.config_pb2 import UltrasonicConfig
from drivers.radar.config_pb2 import RadarConfig
from common.configs.vehicle_config_pb2 import VehicleConfig


def round_half_up(value: float, decimals: int = 1) -> float:
    """
    标准四舍五入函数（0.5向上舍入）
    
    Args:
        value: 要四舍五入的值
        decimals: 保留的小数位数
    
    Returns:
        四舍五入后的值
    """
    if value == 0:
        return 0.0
    
    multiplier = 10 ** decimals
    if value >= 0:
        return math.floor(value * multiplier + 0.5) / multiplier
    else:
        return math.ceil(value * multiplier - 0.5) / multiplier


class VehicleConfigUpdater:
    """车辆配置更新器"""
    
    # ============= 开关配置 =============
    # TODO: 待确认下游是否需要 ultrasonics.cfg 中的 id 字段，确认后可删除此开关
    #KEEP_ULTRASONIC_ID_FIELD = True  # 是否保留 ultrasonics.cfg 中的 id 字段（proto 中未定义）
    KEEP_ULTRASONIC_ID_FIELD = False  # 是否保留 ultrasonics.cfg 中的 id 字段（proto 中未定义）
    
    # TODO: 待更新 Python proto 后可关闭此开关
    KEEP_RADAR_TYPE_FIELD = True  # 是否用正则保留 radars.cfg 中的 type 字段（proto enum 值可能缺失）
    
    # 相机名称映射：Excel中的名称 -> cameras.cfg中的名称
    CAMERA_MAPPING = {
        'Traffic Camera 1': 'traffic_2',
        'Camera 1': 'camera_1',
        'Camera 2': 'camera_2',
        'Camera 3': 'camera_3',
        'Camera 4': 'camera_4',
        'Camera 5': 'camera_5',
        'Camera 6': 'camera_6',
        'Surround Camera 1': 'panoramic_1',
        'Surround Camera 2': 'panoramic_2',
        'Surround Camera 3': 'panoramic_3',
        'Surround Camera 4': 'panoramic_4',
    }
    
    # 超声波传感器厂家 -> frame_id 前缀映射（键名需与 Excel 厂商列一致）
    USS_VENDOR_PREFIX = {
        'VENDOR_A': 'vendor_a_',
        'VENDOR_B': 'vendor_b_',
        'VENDOR_C': 'vendor_c_',
    }
    
    # 超声波传感器名称映射：Excel中的名称 -> 位置后缀
    # F=Front, R=Rear, L=Left, R=Right, S=Side, M=Middle
    USS_POSITION_SUFFIX = {
        'USS3/USS-FRS': 'fsr',   # Front Right Side
        'USS2/USS-FR': 'fr',     # Front Right
        'USS1/USS-FRM': 'fmr',   # Front Right Middle
        'USS12/USS-FLM': 'fml',  # Front Left Middle
        'USS11/USS-FL': 'fl',    # Front Left
        'USS10/USS-FLS': 'fsl',  # Front Left Side
        'USS4/USS-RRS': 'rsr',   # Rear Right Side
        'USS5/USS-RR': 'rr',     # Rear Right
        'USS6/USS-RRM': 'rmr',   # Rear Right Middle
        'USS7/USS-RLM': 'rml',   # Rear Left Middle
        'USS8/USS-RL': 'rl',     # Rear Left
        'USS9/USS-RLS': 'rsl',   # Rear Left Side
    }
    
    # 超声波传感器名称映射：Excel 名称 -> ultrasonics.cfg 中可能的 frame_id 列表
    USS_MAPPING = {
        'USS3/USS-FRS': ['vendor_a_fsr', 'vendor_b_fsr', 'vendor_c_fsr'],
        'USS2/USS-FR': ['vendor_a_fr', 'vendor_b_fr', 'vendor_c_fr'],
        'USS1/USS-FRM': ['vendor_a_fmr', 'vendor_b_fmr', 'vendor_c_fmr'],
        'USS12/USS-FLM': ['vendor_a_fml', 'vendor_b_fml', 'vendor_c_fml'],
        'USS11/USS-FL': ['vendor_a_fl', 'vendor_b_fl', 'vendor_c_fl'],
        'USS10/USS-FLS': ['vendor_a_fsl', 'vendor_b_fsl', 'vendor_c_fsl'],
        'USS4/USS-RRS': ['vendor_a_rsr', 'vendor_b_rsr', 'vendor_c_rsr'],
        'USS5/USS-RR': ['vendor_a_rr', 'vendor_b_rr', 'vendor_c_rr'],
        'USS6/USS-RRM': ['vendor_a_rmr', 'vendor_b_rmr', 'vendor_c_rmr'],
        'USS7/USS-RLM': ['vendor_a_rml', 'vendor_b_rml', 'vendor_c_rml'],
        'USS8/USS-RL': ['vendor_a_rl', 'vendor_b_rl', 'vendor_c_rl'],
        'USS9/USS-RLS': ['vendor_a_rsl', 'vendor_b_rsl', 'vendor_c_rsl'],
    }
    
    # 雷达名称映射：Excel中的名称 -> radars.cfg中的frame_id
    RADAR_MAPPING = {
        'MRR/LRR': 'mrr_1',
        'SRR3/SRR-RL': 'srr_1',
        'SRR2/SRR-RR': 'srr_2',
        'SRR4/SRR-FL': 'srr_3',
        'SRR1/SRR-FR': 'srr_4',
    }
    
    # ============= 每个雷达的属性配置（按 frame_id） =============
    # coord_type: 'vehicle_coordinate'=车辆坐标系, 'radar_coordinate'=雷达自身坐标系
    # radar_type: 'normal_radar'=普通雷达, '4d_radar'=4D雷达
    #
    # 目前共有3种属性配置的radar:
    #   vehicle_coordinate + normal_radar: x=0, y=0, z=0, qw=1
    #   radar_coordinate + normal_radar:   x=Excel, y=Excel, z=0, qw=1
    #   vehicle_coordinate + 4d_radar:       x=0, y=0, z=0, qw=1
    #
    ## 整车雷达配置1：
    # 主毫米波雷达：车辆坐标系，普通雷达
    # 角雷达1：车辆坐标系，普通雷达
    # 角雷达2：雷达坐标系，普通雷达
    RADAR_PROPERTIES1 = {
        'mrr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_2': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
    }
    ## 整车雷达配置2：
    # 主毫米波雷达：雷达坐标系，普通雷达
    # 角雷达1：车辆坐标系，普通雷达
    # 角雷达2：车辆坐标系，普通雷达 
    RADAR_PROPERTIES2 = {
        'mrr_1': {'coord_type': 'radar_coordinate', 'radar_type': 'normal_radar'},
        'srr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_2': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
    }
    ## 整车雷达配置3：
    # 主毫米波雷达：雷达坐标系，4D雷达
    # 角雷达1：车辆坐标系，普通雷达
    # 角雷达2：车辆坐标系，普通雷达
    # 角雷达3：车辆坐标系，普通雷达
    # 角雷达4：车辆坐标系，普通雷达
    RADAR_PROPERTIES3 = {
        'mrr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': '4d_radar'},
        'srr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_2': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
    }
    RADAR_PROPERTIES4 = {
        'mrr_1': {'coord_type': 'radar_coordinate', 'radar_type': 'normal_radar'},
        'srr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_2': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_3': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_4': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
    }
    RADAR_PROPERTIES5 = {
        'mrr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_1': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_2': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_3': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
        'srr_4': {'coord_type': 'vehicle_coordinate', 'radar_type': 'normal_radar'},
    }
    
    # ============= 选择当前使用的雷达配置 =============
    @staticmethod
    def count_radars_in_cfg(raw_config_dir: str) -> int:
        """读取 raw_config 下的 radars.cfg，统计 radar 数量"""
        radars_cfg_path = Path(raw_config_dir) / "radars.cfg"
        if not radars_cfg_path.exists():
            return 0
        with open(radars_cfg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 统计 config { 块的数量
        import re
        return len(re.findall(r'\bconfig\s*\{', content))
    
    @staticmethod
    def select_radar_properties(main_radar_type: str, raw_config_dir: str = None) -> Dict[str, Dict]:
        """
        根据主雷达类型和raw_config中的雷达数量，选择对应的整车雷达配置。
        
        Args:
            main_radar_type: 主雷达类型 ('normal_radar' 或 '4d_radar')
            raw_config_dir: 原始配置目录路径，用于读取radars.cfg判断雷达数量
        """
        # 统计 raw_config 中的雷达数量
        radar_count = 0
        if raw_config_dir:
            radar_count = VehicleConfigUpdater.count_radars_in_cfg(raw_config_dir)
            print(f"  检测到 raw_config 中有 {radar_count} 个 Radar 配置")
        
        # 判断车厂类型（根据 raw_config 目录名）
        is_radar_frame_variant = False
        is_oem_e_variant = False
        if raw_config_dir:
            raw_config_name = Path(raw_config_dir).name.lower()
            is_radar_frame_variant = (
                'oem_c' in raw_config_name or 'oem_d' in raw_config_name)
            is_oem_e_variant = 'oem_e' in raw_config_name
        
        if main_radar_type == "normal_radar":
            if radar_count == 5:
                if is_oem_e_variant:
                    print(f"  → 选择 5 Radar + 普通主雷达 配置 [OEM-E] (RADAR_PROPERTIES5)")
                    return VehicleConfigUpdater.RADAR_PROPERTIES5
                else:
                    print(f"  → 选择 5 Radar + 普通主雷达 配置 [OEM-D] (RADAR_PROPERTIES4)")
                    return VehicleConfigUpdater.RADAR_PROPERTIES4
            elif is_radar_frame_variant:
                print(f"  → 选择 3 Radar + 普通主雷达 配置 [OEM-C] (RADAR_PROPERTIES2)")
                return VehicleConfigUpdater.RADAR_PROPERTIES2
            else:
                print(f"  → 选择 3 Radar + 普通主雷达 配置 (RADAR_PROPERTIES1)")
                return VehicleConfigUpdater.RADAR_PROPERTIES1
        elif main_radar_type == "4d_radar":
            print(f"  → 选择 3 Radar + 4D主雷达 配置 (RADAR_PROPERTIES3)")
            return VehicleConfigUpdater.RADAR_PROPERTIES3
        else:
            raise ValueError(f"不支持的雷达配置: main_radar_type={main_radar_type} 请检查输入参数")

    @staticmethod
    def check_excel_has_model_column(excel_path: str) -> bool:
        """检查 Excel 是否有传感器型号列"""
        try:
            df = pd.read_excel(excel_path, sheet_name=2, header=None)
            
            # 查找数据起始行
            header_row = None
            for idx, row in df.iterrows():
                row_str = ' '.join([str(val) for val in row.values if pd.notna(val)])
                if '序号' in row_str and '传感器' in row_str:
                    header_row = idx
                    break
            
            if header_row is None:
                return False
            
            col_model = 8  # 传感器型号列
            col_name_row = header_row + 1
            
            if len(df.columns) > col_model:
                header_val = str(df.iloc[header_row, col_model]) if pd.notna(df.iloc[header_row, col_model]) else ''
                col_name_val = str(df.iloc[col_name_row, col_model]) if pd.notna(df.iloc[col_name_row, col_model]) else ''
                if '型号' in header_val or 'Model' in header_val or '型号' in col_name_val or 'Model' in col_name_val:
                    return True
            return False
        except Exception:
            return False

    def __init__(self, excel_path: str, raw_config_dir: str, output_config_dir: str, radar_properties: Dict[str, Dict]):
        """
        初始化
        
        Args:
            excel_path: Excel文件路径
            raw_config_dir: 原始配置目录路径
            output_config_dir: 输出配置目录路径
            radar_properties: 雷达属性配置
        """
        self.excel_path = Path(excel_path)
        self.raw_config_dir = Path(raw_config_dir)
        self.output_config_dir = Path(output_config_dir)
        self.radar_properties = radar_properties
        self.sensor_data = {}
        
    def load_excel_data(self) -> Dict[str, Dict]:
        """
        从Excel文件中加载所有传感器数据
        
        Returns:
            传感器数据字典，key为传感器名称
        """
        try:
            # 读取第三个sheet（索引为2）
            df = pd.read_excel(self.excel_path, sheet_name=2, header=None)
            
            # 查找数据起始行（包含"序号"和"传感器"的行）
            header_row = None
            for idx, row in df.iterrows():
                row_str = ' '.join([str(val) for val in row.values if pd.notna(val)])
                if '序号' in row_str and '传感器' in row_str:
                    header_row = idx
                    break
            
            if header_row is None:
                raise ValueError("无法找到Excel数据表头")
            
            print(f"Excel表中传感器数据起始行: {header_row}")
            # 列名行在header_row + 1
            col_name_row = header_row + 1
            # 数据行从header_row + 2开始（跳过表头行和列名行）
            data_start = header_row + 2 #Lidar1所在行
            
            # 提取列索引
            col_no = 0
            col_sensor = 1
            col_x = 2
            col_y = 3
            col_z = 4
            col_yaw = 5  # ψ
            col_pitch = 6  # θ
            col_roll = 7  # φ
            col_model = 8  # 传感器型号（可选列）
            
            # 验证列名行
            col_names = [str(df.iloc[col_name_row, i]).strip() if pd.notna(df.iloc[col_name_row, i]) else '' 
                        for i in range(len(df.columns))]
            print(f"数据列名: {col_names}")
            
            # 检查是否有传感器型号列
            has_model_col = False
            if len(df.columns) > col_model:
                # 检查列名或表头是否包含"型号"或"Model"
                header_val = str(df.iloc[header_row, col_model]) if pd.notna(df.iloc[header_row, col_model]) else ''
                col_name_val = str(df.iloc[col_name_row, col_model]) if pd.notna(df.iloc[col_name_row, col_model]) else ''
                if '型号' in header_val or 'Model' in header_val or '型号' in col_name_val or 'Model' in col_name_val:
                    has_model_col = True
                    print(f"✓ 检测到传感器型号列")
            
            if not has_model_col:
                print(f"⚠ 此Excel没有传感器型号列，需要人工确认 raw_config 中的uss frame_id 是否正确！")
            
            # 读取所有传感器数据
            sensor_data = {}
            # 打印表头
            if has_model_col:
                print(f"  {'NO':<4} {'传感器名称':<22} {'X':>8}  {'Y':>8}  {'Z':>8}  |  {'qx':>7}  {'qy':>7}  {'qz':>7}  | {'厂家':<8} {'型号':<15}")
                print(f"  {'-'*4} {'-'*22} {'-'*8}  {'-'*8}  {'-'*8}  |  {'-'*7}  {'-'*7}  {'-'*7}  | {'-'*8} {'-'*15}")
            else:
                print(f"  {'NO':<4} {'传感器名称':<22} {'X':>8}  {'Y':>8}  {'Z':>8}  |  {'qx':>7}  {'qy':>7}  {'qz':>7}")
                print(f"  {'-'*4} {'-'*22} {'-'*8}  {'-'*8}  {'-'*8}  |  {'-'*7}  {'-'*7}  {'-'*7}")
            for idx in range(data_start, len(df)):
                row = df.iloc[idx]
                
                # 检查是否到达数据末尾
                sensor_name = str(row[col_sensor]) if pd.notna(row[col_sensor]) else ""
                # 检查传感器名称是否为空，或者为空字符串，或者为nan
                if not sensor_name or sensor_name.strip() == "" or sensor_name.lower() == "nan":
                    raise ValueError(f"第 {idx+1} 行传感器名称为空，请检查Excel文件，退出程序")
                
                # 提取传感器名称（可能包含换行符，取第一部分）
                sensor_name_clean = sensor_name.split('\n')[0].strip()
                
                # 提取位置和方向数据
                try:
                    # 检查是否为无效值（/, N/A, NA, 空等），跳过未安装的传感器
                    def is_invalid(val):
                        if pd.isna(val):
                            return True
                        val_str = str(val).strip().upper()
                        # 检查是否为无效值
                        if val_str in ['/', 'N/A', 'NA', '-', '--', '', '无', 'NONE', 'NULL']:
                            return True
                        # 尝试转换为数字，如果失败也视为无效
                        try:
                            float(val)
                            return False
                        except (ValueError, TypeError):
                            return True
                    
                    if any(is_invalid(row[col]) for col in [col_x, col_y, col_z, col_yaw, col_pitch, col_roll]):
                        no = idx - data_start + 1
                        print(f"  {no:>2}. {sensor_name_clean:<22} [跳过 - 传感器未安装]")
                        continue  # 跳过未安装的传感器
                    
                    # 将值转换为float，如果值为空或无效，排除ValueError异常，退出程序。
                    x = float(row[col_x]) if pd.notna(row[col_x]) else None
                    y = float(row[col_y]) if pd.notna(row[col_y]) else None
                    z = float(row[col_z]) if pd.notna(row[col_z]) else None
                    yaw = float(row[col_yaw]) if pd.notna(row[col_yaw]) else None
                    pitch = float(row[col_pitch]) if pd.notna(row[col_pitch]) else None
                    roll = float(row[col_roll]) if pd.notna(row[col_roll]) else None
                    
                    # 所有数值四舍五入到小数点后一位（标准四舍五入，0.5向上）
                    x = round_half_up(x, 1) if x is not None else None
                    y = round_half_up(y, 1) if y is not None else None
                    z = round_half_up(z, 1) if z is not None else None
                    yaw = round_half_up(yaw, 1) if yaw is not None else None
                    pitch = round_half_up(pitch, 1) if pitch is not None else None
                    roll = round_half_up(roll, 1) if roll is not None else None
                    
                    # 解析传感器型号（厂家_具体型号）
                    vendor = None
                    model = None
                    if has_model_col:
                        model_val = str(row[col_model]).strip() if pd.notna(row[col_model]) else ''
                        if model_val and model_val.upper() not in ['N/A', 'NA', '/', '-', '']:
                            parts = model_val.split('_', 1)
                            if len(parts) >= 2:
                                vendor = parts[0].upper()  # 厂家名转大写
                                model = parts[1]
                            else:
                                vendor = model_val.upper()
                                model = ''
                    
                    sensor_data[sensor_name_clean] = {
                        'x': x, 'y': y, 'z': z,
                        'qx': roll, 'qy': pitch, 'qz': yaw, 'qw': 0,
                        'vendor': vendor,  # 厂家
                        'model': model      # 具体型号
                    }
                    # 格式化输出：序号、传感器名、位置xyz、姿态qxqyqz、厂家型号
                    no = idx - data_start + 1
                    if has_model_col:
                        vendor_str = vendor if vendor else '-'
                        model_str = model if model else '-'
                        print(f"  {no:>2}. {sensor_name_clean:<22} x={x:>8.1f}  y={y:>8.1f}  z={z:>8.1f}  |  qx={roll:>7.1f}  qy={pitch:>7.1f}  qz={yaw:>7.1f}  | {vendor_str:<8} {model_str:<15}")
                    else:
                        print(f"  {no:>2}. {sensor_name_clean:<22} x={x:>8.1f}  y={y:>8.1f}  z={z:>8.1f}  |  qx={roll:>7.1f}  qy={pitch:>7.1f}  qz={yaw:>7.1f}")
                    
                except (ValueError, TypeError) as e:
                    # 提取float值时数据解析异常，打印详细信息
                    print(f"  错误行数据: x={row[col_x]}, y={row[col_y]}, z={row[col_z]}, yaw={row[col_yaw]}, pitch={row[col_pitch]}, roll={row[col_roll]}")
                    raise ValueError(f"第 {idx+1} 行 [{sensor_name_clean}] 数据异常，无法转换为数字，请检查Excel文件") from None
        except ValueError:
            raise  # ValueError 直接抛出，保留原始错误信息
        except Exception as e:
            raise RuntimeError(f"读取Excel文件失败: {e}")
        print(f"成功从Excel加载 {len(sensor_data)} 个传感器的数据")
        print(sensor_data.keys())
        return sensor_data
    
    def update_cameras_cfg(self, sensor_data: Dict[str, Dict]):
        """使用 Proto 更新 cameras.cfg 文件"""
        config_path = self.raw_config_dir / 'cameras.cfg'
        output_path = self.output_config_dir / 'cameras.cfg'
        
        if not config_path.exists():
            raise FileNotFoundError(f"cameras.cfg 不存在: {config_path}")
        
        print(f"\n更新 cameras.cfg (使用 Proto)...")
        
        # 使用 Proto 读取 cfg 文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        camera_configs = CameraConfigs()
        text_format.Parse(content, camera_configs)
        
        # 创建反向映射: cfg名称 -> Excel名称
        cfg_to_excel = {v: k for k, v in self.CAMERA_MAPPING.items()}
        
        # 更新每个相机的外参
        for config in camera_configs.config:
            camera_name = config.camera_dev
            
            # 查找对应的 Excel 名称
            excel_name = cfg_to_excel.get(camera_name)
            if excel_name is None:
                raise ValueError(f"相机 {camera_name} 未在 CAMERA_MAPPING 映射表中，请检查映射配置")
            
            # 在 excel中提取的 sensor_data 中查找数据
            data = sensor_data.get(excel_name)
            if data is None:
                raise ValueError(f"Excel中未找到 {excel_name} 的数据，请检查Excel文件")
            
            # 使用 Proto API 更新 sensor_to_cam
            sensor_to_cam = config.parameters.extrinsic.sensor_to_cam
            sensor_to_cam.position.x = data['x']
            sensor_to_cam.position.y = data['y']
            sensor_to_cam.position.z = data['z']
            sensor_to_cam.orientation.qx = data['qx']
            sensor_to_cam.orientation.qy = data['qy']
            sensor_to_cam.orientation.qz = data['qz']
            sensor_to_cam.orientation.qw = data['qw']
            
            print(f"  已更新: {camera_name} ← {excel_name}")
        
        # 使用 Proto 保存 cfg 文件
        self.output_config_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_format.MessageToString(camera_configs, float_format='.9g'))
        
        print(f"  cameras.cfg 已保存: {output_path}")
    
    def update_lidars_cfg(self, sensor_data: Dict[str, Dict], car_config_path: Path):
        """使用 Proto 更新 lidars.cfg 文件"""
        config_path = self.raw_config_dir / 'lidars.cfg'
        output_path = self.output_config_dir / 'lidars.cfg'
        lidar_node_cfg_path = self.raw_config_dir / 'lidar_node.cfg'
        
        if not config_path.exists():
            raise FileNotFoundError(f"lidars.cfg 不存在: {config_path}")
        
        print(f"\n更新 lidars.cfg (使用 Proto)...")
        
        # 从 lidar_node.cfg 读取 frame_id 和 type
        # 从 lidar_node.cfg 读取所有 lidar 的 frame_id 和 type（支持多个 lidar）
        lidar_node_list = []  # [(frame_id, type), ...] 按顺序保存
        print(f"  尝试读取 lidar_node.cfg: {lidar_node_cfg_path}")
        if lidar_node_cfg_path.exists():
            with open(lidar_node_cfg_path, 'r', encoding='utf-8') as f:
                lidar_node_content = f.read()
            
            # 使用括号匹配提取每个 config 块（处理深度嵌套）
            i = 0
            while True:
                # 找到 "config {" 的位置
                match = re.search(r'config\s*\{', lidar_node_content[i:])
                if not match:
                    break
                start = i + match.start()
                # 从 { 开始计数括号
                brace_start = i + match.end() - 1  # { 的位置
                brace_count = 1
                j = brace_start + 1
                while j < len(lidar_node_content) and brace_count > 0:
                    if lidar_node_content[j] == '{':
                        brace_count += 1
                    elif lidar_node_content[j] == '}':
                        brace_count -= 1
                    j += 1
                
                if brace_count == 0:
                    block = lidar_node_content[brace_start:j]
                    # 从块中提取 frame_id 和 type
                    frame_id_match = re.search(r'frame_id:\s*"([^"]+)"', block)
                    type_match = re.search(r'type:\s*(\w+)', block)
                    if frame_id_match and type_match:
                        frame_id = frame_id_match.group(1)
                        lidar_type = type_match.group(1)
                        lidar_node_list.append((frame_id, lidar_type))
                        print(f"  从 lidar_node.cfg 读取: frame_id={frame_id}, type={lidar_type}")
                
                i = j
            
            if not lidar_node_list:
                print(f"  警告: lidar_node.cfg 中未找到有效的 lidar 配置")
            else:
                print(f"  共找到 {len(lidar_node_list)} 个 lidar 配置")
        else:
            print(f"  警告: lidar_node.cfg 不存在: {lidar_node_cfg_path}")
        
        # 从 car_config.cfg 读取 vehicle_to_sensing_translation (使用 Proto)
        vehicle_to_sensing = None
        if car_config_path.exists():
            with open(car_config_path, 'r', encoding='utf-8') as f:
                car_content = f.read()
            
            car_config = VehicleConfig()
            text_format.Parse(car_content, car_config, allow_unknown_field=True)
            
            if car_config.sensors_parameters.HasField('vehicle_to_sensing_translation'):
                trans = car_config.sensors_parameters.vehicle_to_sensing_translation
                vehicle_to_sensing = {
                    'x': trans.x,
                    'y': trans.y,
                    'z': trans.z
                }
                print(f"  从 car_config.cfg 读取 vehicle_to_sensing: x={vehicle_to_sensing['x']}, y={vehicle_to_sensing['y']}, z={vehicle_to_sensing['z']}")
            else:
                print(f"  警告: car_config.cfg 中未找到 vehicle_to_sensing_translation")
        else:
            print(f"  警告: car_config.cfg 不存在: {car_config_path}")
        
        # 使用 Proto 读取 cfg 文件（只读取 vehicle_to_sensing 部分作为模板）
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lidar_config = LidarConfig()
        text_format.Parse(content, lidar_config)
        
        # frame_id 后缀 _20X 与 Excel LiDAR 名称的映射
        LIDAR_SUFFIX_MAPPING = {
            '201': 'NLiDAR 4',
            '202': 'LiDAR 1',
            '203': 'NLiDAR 2',
            '204': 'NLiDAR 1',
            '205': 'NLiDAR 3',
            '206': 'FLiDAR 1',
            '207': 'FLiDAR 3',
            '208': 'FLiDAR 2',
        }
        
        # 查找所有 LiDAR 数据（FLiDAR 1, FLiDAR 2, NLiDAR 1 等）
        lidar_data_dict = {}  # {excel_name: data}
        for key in sensor_data.keys():
            if 'LiDAR' in key or 'lidar' in key.lower():
                data = sensor_data[key]
                if data.get('x') is not None:  # 只收集有有效数据的 LiDAR
                    # 标准化名称（去掉多余空格）
                    normalized_key = re.sub(r'\s+', ' ', key).strip()
                    lidar_data_dict[normalized_key] = data
                    print(f"  找到激光雷达数据: {key}")
        
        print(f"  lidar_node.cfg: {len(lidar_node_list)} 个, Excel: {len(lidar_data_dict)} 个")
        
        if len(lidar_node_list) == 0:
            raise ValueError("lidar_node.cfg 中未找到任何 lidar 配置")
        
        # 建立 lidar_node.cfg 与 Excel 数据的对应关系
        # 规则：根据 frame_id 后缀 _20X 映射到 Excel 名称
        lidar_mapping = []  # [(frame_id, lidar_type, excel_name, excel_data), ...]
        
        for frame_id, lidar_type in lidar_node_list:
            # 提取 frame_id 后缀（最后3位数字）
            suffix_match = re.search(r'_?(\d{3})$', frame_id)
            if suffix_match:
                suffix = suffix_match.group(1)
            else:
                # 如果没有3位数字后缀，尝试2位
                suffix_match = re.search(r'_?(\d+)$', frame_id)
                suffix = suffix_match.group(1) if suffix_match else None
            
            excel_name = None
            excel_data = None
            
            if suffix and suffix in LIDAR_SUFFIX_MAPPING:
                target_name = LIDAR_SUFFIX_MAPPING[suffix]
                # 在 sensor_data 中查找匹配的名称
                for key, data in lidar_data_dict.items():
                    # 模糊匹配：去掉空格比较
                    key_normalized = key.replace(' ', '').upper()
                    target_normalized = target_name.replace(' ', '').upper()
                    if target_normalized in key_normalized or key_normalized in target_normalized:
                        excel_name = key
                        excel_data = data
                        break
                
                if excel_name:
                    print(f"  映射(后缀{suffix}): {frame_id} ({lidar_type}) ← {excel_name}")
                else:
                    print(f"  ⚠ 警告: 后缀 {suffix} 对应 {target_name}，但在Excel中未找到该数据")
            else:
                print(f"  ⚠ 警告: frame_id={frame_id} 的后缀无法识别")
            
            if excel_data is None:
                raise ValueError(f"无法为 {frame_id} (后缀:{suffix}) 找到对应的 Excel 数据 (期望: {LIDAR_SUFFIX_MAPPING.get(suffix, '未知')})")
            
            lidar_mapping.append((frame_id, lidar_type, excel_name, excel_data))
        
        # 清空现有 config，根据 lidar_node.cfg 数量动态创建
        del lidar_config.config[:]
        
        for idx, (frame_id, lidar_type, excel_name, lidar_data) in enumerate(lidar_mapping):
            
            # 创建新的 config 块
            config = lidar_config.config.add()
            config.frame_id = frame_id
            
            # 设置 model，如果 proto 中不存在则报错退出
            try:
                config.model = LidarModel.Value(lidar_type)
            except ValueError:
                raise ValueError(f"LiDAR model '{lidar_type}' 在 proto 中不存在，请更新 proto 定义或修改 lidar_node.cfg")
            
            # 设置 sensor_to_lidar
            transform = config.sensor_to_lidar.add()
            transform.position.x = lidar_data['x']
            transform.position.y = lidar_data['y']
            transform.position.z = lidar_data['z']
            
            # orientation 设置为单位四元数
            transform.orientation.qx = 0
            transform.orientation.qy = 0
            transform.orientation.qz = 0
            transform.orientation.qw = 1
            
            # 设置 ring_id_start 和 ring_id_end（按顺序递增）
            config.ring_id_start = idx
            config.ring_id_end = idx + 1
            
            print(f"  已创建: {config.frame_id} (model={lidar_type}) ← {excel_name}")
        
        # 更新 vehicle_to_sensing（从 car_config.cfg 读取）
        if vehicle_to_sensing and lidar_config.vehicle_to_sensing:
            lidar_config.vehicle_to_sensing.position.x = vehicle_to_sensing['x']
            lidar_config.vehicle_to_sensing.position.y = vehicle_to_sensing['y']
            lidar_config.vehicle_to_sensing.position.z = vehicle_to_sensing['z']
            # orientation 保持默认 qw=1
            lidar_config.vehicle_to_sensing.orientation.qx = 0
            lidar_config.vehicle_to_sensing.orientation.qy = 0
            lidar_config.vehicle_to_sensing.orientation.qz = 0
            lidar_config.vehicle_to_sensing.orientation.qw = 1
            print(f"  已更新 vehicle_to_sensing: x={vehicle_to_sensing['x']}, y={vehicle_to_sensing['y']}, z={vehicle_to_sensing['z']}")
        
        # 使用 Proto 保存 cfg 文件
        self.output_config_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text_format.MessageToString(lidar_config, float_format='.9g'))
        
        print(f"  lidars.cfg 已保存: {output_path}")
    
    def update_ultrasonics_cfg(self, sensor_data: Dict[str, Dict]):
        """使用 Proto 更新 ultrasonics.cfg 文件"""
        config_path = self.raw_config_dir / 'ultrasonics.cfg'
        output_path = self.output_config_dir / 'ultrasonics.cfg'
        
        if not config_path.exists():
            raise FileNotFoundError(f"ultrasonics.cfg 不存在: {config_path}")
        
        print(f"\n更新 ultrasonics.cfg (使用 Proto)...")
        
        # 从第一个 USS 传感器获取厂家信息
        uss_vendor = None
        expected_prefix = None
        for excel_name in self.USS_POSITION_SUFFIX.keys():
            for key in sensor_data.keys():
                if excel_name in key:
                    uss_vendor = sensor_data[key].get('vendor')
                    if uss_vendor:
                        expected_prefix = self.USS_VENDOR_PREFIX.get(uss_vendor)
                        print(f"  USS厂家: {uss_vendor} → frame_id前缀: {expected_prefix or '未知'}")
                    break
            if uss_vendor:
                break
        
        if not uss_vendor:
            print(f"  ⚠ Excel中未找到USS厂家信息，将使用raw_config中已有的frame_id")
        
        # 使用 Proto 读取 cfg 文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 先提取 frame_id -> id 映射（id 字段不在 proto 定义中，需要手动保留）
        frame_id_to_id = {}
        if self.KEEP_ULTRASONIC_ID_FIELD:
            for match in re.finditer(r'frame_id:\s*"([^"]+)".*?id:\s*(\d+)', content, re.DOTALL):
                frame_id_to_id[match.group(1)] = int(match.group(2))
        
        ultra_config = UltrasonicConfig()
        text_format.Parse(content, ultra_config, allow_unknown_field=True)
        
        # 检查 raw_config 中的 frame_id 前缀是否与 Excel 厂家匹配，不匹配则自动替换
        actual_prefix = None
        if expected_prefix and ultra_config.config:
            first_frame_id = ultra_config.config[0].frame_id
            if not first_frame_id.startswith(expected_prefix):
                actual_prefix = first_frame_id.split('_')[0] + '_' if '_' in first_frame_id else None
                print(f"  🔄 检测到frame_id前缀不匹配: raw_config({actual_prefix}) vs Excel({expected_prefix})")
                print(f"     将自动把 {actual_prefix} 替换为 {expected_prefix}")
                # 替换所有 USS 配置的 frame_id 前缀
                for config in ultra_config.config:
                    if config.frame_id.startswith(actual_prefix):
                        old_frame_id = config.frame_id
                        new_frame_id = expected_prefix + config.frame_id[len(actual_prefix):]
                        config.frame_id = new_frame_id
                        # 同步更新 frame_id_to_id 映射
                        if old_frame_id in frame_id_to_id:
                            frame_id_to_id[new_frame_id] = frame_id_to_id.pop(old_frame_id)
        
        # 更新每个超声波传感器
        for config in ultra_config.config:
            frame_id = config.frame_id
            
            # 查找对应的 Excel 名称
            excel_name = None
            for excel_n, cfg_ids in self.USS_MAPPING.items():
                if frame_id in cfg_ids:
                    excel_name = excel_n
                    break
            
            if excel_name is None:
                continue
            
            # 在 sensor_data 中查找数据
            data = None
            for key in sensor_data.keys():
                if excel_name in key:
                    data = sensor_data[key]
                    break
            
            if data is None:
                raise ValueError(f"Excel中未找到 {frame_id} ({excel_name}) 的超声波数据，请检查Excel文件")
            
            # 使用 Proto API 更新 sensor_to_ultrasonic
            transform = config.sensor_to_ultrasonic
            transform.position.x = data['x']
            transform.position.y = data['y']
            transform.position.z = data['z']
            transform.orientation.qx = data['qx']
            transform.orientation.qy = data['qy']
            transform.orientation.qz = data['qz']
            transform.orientation.qw = data['qw']
            
            print(f"  已更新: {frame_id} ← {excel_name}")
        
        # 使用 Proto 保存 cfg 文件
        self.output_config_dir.mkdir(parents=True, exist_ok=True)
        output_content = text_format.MessageToString(ultra_config, float_format='.9g')
        
        # 把 id 字段加回去（在每个 config 块的末尾，} 之前插入 id）
        if frame_id_to_id:
            for frame_id, id_val in frame_id_to_id.items():
                # 在 frame_id: "xxx" 对应的 config 块末尾加入 id
                pattern = rf'(frame_id:\s*"{frame_id}".*?)((\n\}}))' 
                replacement = rf'\1\n  id: {id_val}\2'
                output_content = re.sub(pattern, replacement, output_content, flags=re.DOTALL)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"  ultrasonics.cfg 已保存: {output_path}")
    
    def update_navigation_devices_cfg(self, sensor_data: Dict[str, Dict]):
        """更新navigation_devices.cfg文件"""
        config_path = self.raw_config_dir / 'navigation_devices.cfg'
        output_path = self.output_config_dir / 'navigation_devices.cfg'
        
        if not config_path.exists():
            print(f"警告: {config_path} 不存在，跳过")
            return
        
        print(f"\n更新 navigation_devices.cfg...")
        
        # 查找GNSS天线数据
        gnss_data = None
        for key in sensor_data.keys():
            if 'GNSS' in key.upper() or 'ANT' in key.upper() or '天线' in key:
                gnss_data = sensor_data[key]
                print(f"  找到GNSS天线数据: {key}")
                break
        
        if gnss_data is None:
            raise ValueError("Excel中未找到GNSS天线数据，请检查Excel文件")
        
        # 读取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 更新level_arm_imu_to_gnss_antenna_1
        # 注意：Excel中的单位是mm（精确到1位小数），需要转换为m
        # 保持Excel精度：1位小数(mm) → 4位小数(m)
        x_m = round_half_up(gnss_data['x'] / 1000.0, 4)
        y_m = round_half_up(gnss_data['y'] / 1000.0, 4)
        z_m = round_half_up(gnss_data['z'] / 1000.0, 4)
        
        content = re.sub(
            r'level_arm_imu_to_gnss_antenna_1\s*\{\s*x:\s*[\d.-]+\s*y:\s*[\d.-]+\s*z:\s*[\d.-]+',
            f'level_arm_imu_to_gnss_antenna_1 {{\n      x: {x_m}\n      y: {y_m}\n      z: {z_m}',
            content,
            flags=re.DOTALL
        )
        
        self.output_config_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"  navigation_devices.cfg 已保存到: {output_path}")
        print(f"  GNSS天线位置 (m): x={x_m}, y={y_m}, z={z_m}")
    
    def update_radars_cfg(self, sensor_data: Dict[str, Dict]):
        """
        使用 Proto 更新 radars.cfg 文件
        根据雷达类型和坐标系类型采用不同的更新策略:
        - 普通雷达 + 车辆坐标系: x=0, y=0, z=0, qw=1
        - 4D雷达 + 车辆坐标系: x=0, y=0, z=0, qw=1
        - 普通雷达 + 自身坐标系: x=Excel, y=Excel, z=0, qw=1
        - 4D雷达 + 自身坐标系: x=Excel, y=Excel, z=Excel, qw=1
        """
        config_path = self.raw_config_dir / 'radars.cfg'
        output_path = self.output_config_dir / 'radars.cfg'
        
        if not config_path.exists():
            raise FileNotFoundError(f"radars.cfg 不存在: {config_path}")
        
        print(f"\n更新 radars.cfg (使用 Proto)...")
        
        # 使用 Proto 读取 cfg 文件
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计 raw_config 中的 radar 数量
        raw_config_radar_count = len(re.findall(r'\bconfig\s*\{', content))
        
        # 统计 Excel 中有效的 radar 数量（坐标非空）
        excel_valid_radars = []
        for excel_name, frame_id in self.RADAR_MAPPING.items():
            data = sensor_data.get(excel_name)
            if data is not None and data.get('x') is not None:
                excel_valid_radars.append(excel_name)
        excel_radar_count = len(excel_valid_radars)
        
        # 比较并打印
        print(f"  raw_config 中有 {raw_config_radar_count} 个 Radar，Excel 中有 {excel_radar_count} 个有效 Radar")
        if raw_config_radar_count != excel_radar_count:
            raise ValueError(
                f"raw_config ({raw_config_radar_count}个Radar) 与 Excel "
                f"({excel_radar_count}个有效Radar) 数量不一致！请检查车厂是否正确")
        
        # 先提取 frame_id -> type 映射（type 字段的 enum 值可能不在 Python proto 中，需要手动保留）
        frame_id_to_type = {}
        if self.KEEP_RADAR_TYPE_FIELD:
            for match in re.finditer(r'config\s*\{[^}]*frame_id:\s*"([^"]+)"[^}]*type:\s*(\w+)', content, re.DOTALL):
                frame_id_to_type[match.group(1)] = match.group(2)
        
        radar_config = RadarConfig()
        text_format.Parse(content, radar_config, allow_unknown_field=True)
        
        # 创建反向映射: cfg frame_id -> Excel名称
        cfg_to_excel = {v: k for k, v in self.RADAR_MAPPING.items()}
        
        # 记录已更新的雷达
        updated_radars = []
        
        # 更新每个雷达（必须处理所有雷达，不能漏掉）
        for config in radar_config.config:
            frame_id = config.frame_id
            
            # 查找对应的 Excel 名称
            excel_name = cfg_to_excel.get(frame_id)
            if excel_name is None:
                raise ValueError(f"雷达 {frame_id} 未在 RADAR_MAPPING 中定义，请更新 RADAR_MAPPING")
            
            # 在 Excel 提取的 sensor_data 中查找数据
            data = sensor_data.get(excel_name)
            if data is None or data.get('x') is None:
                raise ValueError(f"Excel中未找到 {excel_name} ({frame_id}) 的有效数据，请检查Excel文件")
            
            # 获取该雷达的属性配置
            props = self.radar_properties.get(frame_id)
            if props is None:
                raise ValueError(f"雷达 {frame_id} 未在 radar_properties 中配置，请检查主雷达类型选择是否正确")
            coord_type = props['coord_type']
            radar_type = props['radar_type']
            
            # 根据坐标系类型和雷达类型设置不同的值
            if coord_type == 'vehicle_coordinate':
                # 车辆坐标系（普通雷达和4D雷达都一样）: 全部置零
                config.sensor_to_radar.position.x = 0
                config.sensor_to_radar.position.y = 0
                config.sensor_to_radar.position.z = 0
                print(f"  已更新: {frame_id} ← {excel_name} (车辆坐标系, {radar_type}) [x=0, y=0, z=0]")
            elif coord_type == 'radar_coordinate':
                if radar_type == '4d_radar':
                    # 4D雷达 + 雷达坐标系: 使用 Excel 的 x, y, z
                    config.sensor_to_radar.position.x = data['x']
                    config.sensor_to_radar.position.y = data['y']
                    config.sensor_to_radar.position.z = data['z']
                    print(f"  已更新: {frame_id} ← {excel_name} (雷达坐标系, 4D) [x={data['x']}, y={data['y']}, z={data['z']}]")
                else:
                    # 普通雷达 + 雷达坐标系: 使用 Excel 的 x, y，z=0
                    config.sensor_to_radar.position.x = data['x']
                    config.sensor_to_radar.position.y = data['y']
                    config.sensor_to_radar.position.z = 0
                    print(f"  已更新: {frame_id} ← {excel_name} (雷达坐标系, normal) [x={data['x']}, y={data['y']}, z=0]")
            
            # orientation 统一设置为 qw=1
            config.sensor_to_radar.orientation.qx = 0
            config.sensor_to_radar.orientation.qy = 0
            config.sensor_to_radar.orientation.qz = 0
            config.sensor_to_radar.orientation.qw = 1
            
            # 如果是 4D 雷达，设置 type 为 WHST_STA77_7S_4D_FR
            if radar_type == '4d_radar':
                frame_id_to_type[frame_id] = 'WHST_STA77_7S_4D_FR'
            
            updated_radars.append(frame_id)
        
        # 验证所有 radar_properties 中的雷达都被更新了
        expected_radars = set(self.radar_properties.keys())
        updated_set = set(updated_radars)
        missing_radars = expected_radars - updated_set
        if missing_radars:
            raise ValueError(f"以下雷达未被更新: {missing_radars}，请检查配置")
        
        print(f"  ✓ 已成功更新 {len(updated_radars)} 个雷达配置")
        
        # 使用 Proto 保存 cfg 文件
        self.output_config_dir.mkdir(parents=True, exist_ok=True)
        output_content = text_format.MessageToString(radar_config, float_format='.9g')
        
        # 把 type 字段加回去（在每个 config 块的 frame_id 之后插入 type）
        if frame_id_to_type:
            for frame_id, type_val in frame_id_to_type.items():
                # 在 frame_id: "xxx" 后面插入 type: XXX
                pattern = rf'(frame_id:\s*"{frame_id}")'
                replacement = rf'\1\n  type: {type_val}'
                output_content = re.sub(pattern, replacement, output_content)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_content)
        
        print(f"  radars.cfg 已保存: {output_path}")
    
    def copy_unchanged_files(self):
        """复制不需要更新的文件"""
        files_to_copy = ['car_config.cfg']
        
        for filename in files_to_copy:
            src_path = self.raw_config_dir / filename
            dst_path = self.output_config_dir / filename
            
            if src_path.exists():
                self.output_config_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_path)
                print(f"  已复制: {filename}")
            else:
                print(f"  警告: {filename} 不存在，跳过复制")
    
    def run(self):
        """执行完整的更新流程"""
        print("=" * 60)
        print("开始更新车辆配置文件")
        print("=" * 60)
        
        print(f"\n输入目录: {self.raw_config_dir}")
        print(f"输出目录: {self.output_config_dir}")
        print(f"Excel文件: {self.excel_path}")
        
       
        # 加载Excel数据
        print(f"\n1. 读取Excel文件...")
        sensor_data = self.load_excel_data()
        self.sensor_data = sensor_data  # 保存到实例变量，供GUI获取
         # 创建输出目录
        self.output_config_dir.mkdir(parents=True, exist_ok=True)

        # 更新各个配置文件（任意一个失败则退出程序）
        print(f"\n2. 更新配置文件...")
        try:
            self.update_cameras_cfg(sensor_data)
        except Exception as e:
            raise RuntimeError(f"更新 cameras.cfg 失败: {e}") from None
        
        try:
            self.update_lidars_cfg(sensor_data, self.raw_config_dir / 'car_config.cfg')
        except Exception as e:
            raise RuntimeError(f"更新 lidars.cfg 失败: {e}") from None
        
        try:
            self.update_ultrasonics_cfg(sensor_data)
        except Exception as e:
            raise RuntimeError(f"更新 ultrasonics.cfg 失败: {e}") from None
        
        try:
            self.update_navigation_devices_cfg(sensor_data)
        except Exception as e:
            raise RuntimeError(f"更新 navigation_devices.cfg 失败: {e}") from None
        
        try:
            self.update_radars_cfg(sensor_data)
        except Exception as e:
            raise RuntimeError(f"更新 radars.cfg 失败: {e}") from None
        
        # 复制不需要更新的文件
        print(f"\n3. 复制不需要更新的文件...")
        self.copy_unchanged_files()
        
        print("\n" + "=" * 60)
        print("更新完成！")
        print("=" * 60)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='从Excel文件更新车辆配置文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python update_vehicle_config.py \\
    --excel "ME-D02-25115-v1.0_B26A1_B26A2测试车V1.0传感器布置和车辆尺寸参数表 .xlsx" \\
    --raw-config "raw_config" \\
    --output-config "output_config"
        """
    )
    
    parser.add_argument(
        '--excel',
        type=str,
        required=True,
        help='Excel文件路径'
    )
    
    parser.add_argument(
        '--raw-config',
        type=str,
        required=True,
        help='原始配置目录路径'
    )
    
    parser.add_argument(
        '--output-config',
        type=str,
        required=True,
        help='输出配置目录路径'
    )
    
    parser.add_argument(
        '--main-radar-type',
        type=str,
        required=True,
        choices=['normal_radar', '4d_radar'],
        help='主雷达类型: normal_radar 或 4d_radar'
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    excel_path = Path(args.excel)
    if not excel_path.exists():
        print(f"错误: Excel文件不存在: {excel_path}")
        sys.exit(1)
    
    raw_config_dir = Path(args.raw_config)
    if not raw_config_dir.exists():
        print(f"错误: 原始配置目录不存在: {raw_config_dir}")
        sys.exit(1)
    main_radar_type = args.main_radar_type
    radar_properties = VehicleConfigUpdater.select_radar_properties(main_radar_type, str(raw_config_dir))
    # 执行更新
    updater = VehicleConfigUpdater(str(excel_path), str(raw_config_dir), str(args.output_config), radar_properties)
    try:
        updater.run()
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()

