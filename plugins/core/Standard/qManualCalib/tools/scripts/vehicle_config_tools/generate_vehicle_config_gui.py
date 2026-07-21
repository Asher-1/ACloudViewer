#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆配置文件生成工具 - GUI版本
"""

import sys
import os
import subprocess
from pathlib import Path
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QFileDialog, QLineEdit,
    QGroupBox, QMessageBox, QSplitter, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QProcess
from PyQt5.QtGui import QFont
import pyqtgraph.opengl as gl

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_vehicle_config import VehicleConfigUpdater


# ========== 车辆参数 (mm) ==========
CAR_LENGTH = 5000
CAR_WIDTH = 2000
CAR_HEIGHT = 1500
WHEELBASE = 3000
FRONT_OVERHANG = 1000
WHEEL_RADIUS = 350
WHEEL_WIDTH = 225
TRACK_WIDTH = 1700


# ========== 3D 可视化辅助函数 ==========
def create_pyramid_mesh(pos, size=100, color=(1, 0, 0, 1)):
    """创建金字塔网格（代表相机）"""
    x, y, z = pos
    s = size
    h = size * 1.5
    verts = np.array([
        [x-s/2, y-s/2, z], [x+s/2, y-s/2, z],
        [x+s/2, y+s/2, z], [x-s/2, y+s/2, z], [x, y, z+h],
    ])
    faces = np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4],[0,1,2],[0,2,3]])
    colors = np.array([color] * len(faces))
    return gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False, drawEdges=True)


def create_box_mesh(pos, size=(200, 100, 50), color=(0, 1, 0, 1), translucent=False):
    """创建长方体网格（代表LiDAR或车体）"""
    x, y, z = pos
    dx, dy, dz = size[0]/2, size[1]/2, size[2]/2
    verts = np.array([
        [x-dx,y-dy,z-dz],[x+dx,y-dy,z-dz],[x+dx,y+dy,z-dz],[x-dx,y+dy,z-dz],
        [x-dx,y-dy,z+dz],[x+dx,y-dy,z+dz],[x+dx,y+dy,z+dz],[x-dx,y+dy,z+dz],
    ])
    faces = np.array([[0,1,2],[0,2,3],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[2,3,7],[2,7,6],[0,3,7],[0,7,4],[1,2,6],[1,6,5]])
    colors = np.array([color] * len(faces))
    gl_options = 'translucent' if translucent else 'opaque'
    return gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=False, drawEdges=True, glOptions=gl_options)


def create_wheel(cx, cy, cz, radius=350, width=225, segments=24, color=(0.15, 0.15, 0.15, 1)):
    """创建轮子（圆柱体）"""
    verts = []
    faces = []
    for i in range(segments):
        angle = 2 * np.pi * i / segments
        x = cx + radius * np.cos(angle)
        z = cz + radius * np.sin(angle)
        verts.append([x, cy - width/2, z])
        verts.append([x, cy + width/2, z])
    for i in range(segments):
        i0, i1 = i * 2, i * 2 + 1
        i2, i3 = ((i + 1) % segments) * 2, ((i + 1) % segments) * 2 + 1
        faces.append([i0, i2, i3])
        faces.append([i0, i3, i1])
    left_center = len(verts)
    verts.append([cx, cy - width/2, cz])
    right_center = len(verts)
    verts.append([cx, cy + width/2, cz])
    for i in range(segments):
        i0, i2 = i * 2, ((i + 1) % segments) * 2
        faces.append([left_center, i2, i0])
        i1, i3 = i * 2 + 1, ((i + 1) % segments) * 2 + 1
        faces.append([right_center, i1, i3])
    verts = np.array(verts)
    faces = np.array(faces)
    colors = np.array([color] * len(faces))
    return gl.GLMeshItem(vertexes=verts, faces=faces, faceColors=colors, smooth=True, drawEdges=False)


def create_car_body(width, wheelbase, front_overhang, rear_overhang, ground_clearance,
                    front_height=500, body_height=1000, rear_height=600):
    """创建车体（三个透明长方体：前悬、车体、后悬）"""
    items = []
    z_bottom = ground_clearance
    
    # 前悬长方体
    front_x_center = wheelbase + front_overhang / 2
    front_z_center = z_bottom + front_height / 2
    items.append(create_box_mesh((front_x_center, 0, front_z_center), 
                                  size=(front_overhang, width, front_height), 
                                  color=(0.3, 0.5, 0.7, 0.15), translucent=True))
    
    # 车体长方体
    body_x_center = wheelbase / 2
    body_z_center = z_bottom + body_height / 2
    items.append(create_box_mesh((body_x_center, 0, body_z_center), 
                                  size=(wheelbase, width, body_height), 
                                  color=(0.4, 0.5, 0.6, 0.15), translucent=True))
    
    # 后悬长方体
    rear_x_center = -rear_overhang / 2
    rear_z_center = z_bottom + rear_height / 2
    items.append(create_box_mesh((rear_x_center, 0, rear_z_center), 
                                  size=(rear_overhang, width, rear_height), 
                                  color=(0.3, 0.5, 0.7, 0.15), translucent=True))
    
    return items


def create_colored_axis(length=2000):
    """创建RGB坐标轴: X红 Y绿 Z蓝"""
    axes = []
    axes.append(gl.GLLinePlotItem(pos=np.array([[0,0,0],[length,0,0]]), color=(1,0,0,1), width=3))
    axes.append(gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,length,0]]), color=(0,1,0,1), width=3))
    axes.append(gl.GLLinePlotItem(pos=np.array([[0,0,0],[0,0,length]]), color=(0,0,1,1), width=3))
    return axes


class SensorVisualizer:
    """传感器3D可视化管理器"""
    
    def __init__(self, gl_widget):
        self.gl_widget = gl_widget
        self.sensor_items = []
    
    def init_scene(self, front_height=500, body_height=1000, rear_height=600):
        """初始化3D场景：车辆、轮子、坐标轴（原点在后轴中心）"""
        # RGB坐标轴
        for axis in create_colored_axis(2000):
            self.gl_widget.addItem(axis)
        
        # 网格（放在地面）
        grid = gl.GLGridItem()
        grid.scale(300, 300, 1)
        grid.translate(0, 0, -WHEEL_RADIUS)
        self.gl_widget.addItem(grid)
        
        # 车体
        rear_overhang = CAR_LENGTH - WHEELBASE - FRONT_OVERHANG
        for item in create_car_body(CAR_WIDTH, WHEELBASE, FRONT_OVERHANG, rear_overhang, 0,
                                    front_height=front_height, body_height=body_height, rear_height=rear_height):
            self.gl_widget.addItem(item)
        
        # 4个轮子
        for wx, wy in [(0, -TRACK_WIDTH/2), (0, TRACK_WIDTH/2), (WHEELBASE, -TRACK_WIDTH/2), (WHEELBASE, TRACK_WIDTH/2)]:
            self.gl_widget.addItem(create_wheel(wx, wy, 0))
        
        # 示例传感器
        self.add_example_sensors()
    
    def add_example_sensors(self):
        """添加示例传感器"""
        item = create_box_mesh((WHEELBASE/2, 0, 1850), size=(200, 100, 50), color=(0, 0.8, 0, 0.8))
        self.gl_widget.addItem(item)
        self.sensor_items.append(item)
        
        for pos in [(WHEELBASE + 500, 0, 1500), (WHEELBASE + 300, -800, 1200), (WHEELBASE + 300, 800, 1200)]:
            item = create_pyramid_mesh(pos, size=80, color=(0.9, 0.2, 0.2, 0.8))
            self.gl_widget.addItem(item)
            self.sensor_items.append(item)
    
    def update_from_data(self, sensor_data: dict):
        """根据Excel传感器数据更新3D视图"""
        for item in self.sensor_items:
            self.gl_widget.removeItem(item)
        self.sensor_items.clear()
        
        skipped, added = [], []
        
        for name, data in sensor_data.items():
            x, y, z = data.get('x', 0), data.get('y', 0), data.get('z', 0)
            if x is None or y is None or z is None:
                skipped.append(f"{name}: x={x}, y={y}, z={z}")
                continue
            
            name_lower = name.lower()
            
            if 'surround' in name_lower or 'panoramic' in name_lower:
                item = create_pyramid_mesh((x, y, z), size=200, color=(1.0, 0.5, 0.0, 1.0))
            elif 'camera' in name_lower or 'traffic' in name_lower:
                item = create_pyramid_mesh((x, y, z), size=180, color=(1.0, 0.0, 0.0, 1.0))
            elif 'lidar' in name_lower:
                item = create_box_mesh((x, y, z), size=(300, 200, 100), color=(0.0, 1.0, 0.0, 1.0))
            elif 'mrr' in name_lower or 'lrr' in name_lower:
                item = create_box_mesh((x, y, z), size=(300, 200, 100), color=(0.0, 0.0, 1.0, 1.0))
            elif 'srr' in name_lower:
                item = create_box_mesh((x, y, z), size=(250, 150, 80), color=(0.0, 0.5, 1.0, 1.0))
            elif 'uss' in name_lower:
                item = create_pyramid_mesh((x, y, z), size=200, color=(1.0, 1.0, 0.0, 1.0))
            elif 'gnss' in name_lower or 'ant' in name_lower or '天线' in name:
                item = create_box_mesh((x, y, z), size=(150, 150, 100), color=(0.0, 1.0, 1.0, 1.0))
            elif 'imu' in name_lower:
                item = create_box_mesh((x, y, z), size=(120, 120, 80), color=(1.0, 0.0, 1.0, 1.0))
            else:
                item = create_box_mesh((x, y, z), size=(120, 120, 80), color=(1.0, 1.0, 1.0, 1.0))
            
            self.gl_widget.addItem(item)
            self.sensor_items.append(item)
            added.append(f"{name}: ({x:.0f}, {y:.0f}, {z:.0f})")
        
        print(f"\n=== 3D可视化传感器 ===")
        print(f"已添加 {len(added)} 个:")
        for s in added:
            print(f"  ✓ {s}")
        if skipped:
            print(f"跳过 {len(skipped)} 个 (坐标为None):")
            for s in skipped:
                print(f"  ✗ {s}")
        
        return len(self.sensor_items)


class ClickableLineEdit(QLineEdit):
    """可点击的 QLineEdit"""
    clicked = pyqtSignal()
    
    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class DragDropLineEdit(QLineEdit):
    """支持拖拽文件的 QLineEdit"""
    fileDropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet("border: 2px solid #2196F3; background-color: #e3f2fd;")
        else:
            event.ignore()
    
    def dragLeaveEvent(self, event):
        self.setStyleSheet("")
    
    def dropEvent(self, event):
        self.setStyleSheet("")
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            self.setText(file_path)
            self.fileDropped.emit(file_path)


class WorkerThread(QThread):
    """后台工作线程 - Excel转CFG"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    sensor_data_signal = pyqtSignal(dict)  # 传感器数据信号
    warning_signal = pyqtSignal(str)  # 警告信号（用于弹窗）
    
    def __init__(self, excel_path, raw_config, output_config, radar_properties):
        super().__init__()
        self.excel_path = excel_path
        self.raw_config = raw_config
        self.output_config = output_config
        self.radar_properties = radar_properties
    
    def run(self):
        try:
            import io
            from contextlib import redirect_stdout
            
            output = io.StringIO()
            with redirect_stdout(output):
                updater = VehicleConfigUpdater(
                    self.excel_path,
                    self.raw_config,
                    self.output_config,
                    self.radar_properties
                )
                updater.run()
            
            log_content = output.getvalue()
            self.log_signal.emit(log_content)
            
            # 发送传感器数据用于3D可视化
            self.sensor_data_signal.emit(updater.sensor_data)
            self.finished_signal.emit(True, "Excel转CFG完成！")
        except Exception as e:
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, f"生成失败: {str(e)}")


class SensingWorkerThread(QThread):
    """后台工作线程 - 生成sensing坐标系配置"""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)
    
    def __init__(self, input_config_dir, output_configs_dir):
        super().__init__()
        self.input_config_dir = input_config_dir
        self.output_configs_dir = output_configs_dir  # config 目录
        self.node_path = "/path_to/generate_vehicle_config_node"
    
    def run(self):
        try:
            import shutil
            
            # 检查程序是否存在
            if not Path(self.node_path).exists():
                self.finished_signal.emit(False, f"程序不存在: {self.node_path}")
                return
            
            # 创建输出目录
            config_dir = Path(self.output_configs_dir)
            config_dir.mkdir(parents=True, exist_ok=True)
            
            # model 目录 = config 目录的兄弟目录
            model_dir = config_dir.parent / "model"
            model_dir.mkdir(parents=True, exist_ok=True)
            
            self.log_signal.emit(f"调用: {self.node_path}")
            self.log_signal.emit(f"  输入目录: {self.input_config_dir}")
            self.log_signal.emit(f"  输出目录: {self.output_configs_dir}")
            self.log_signal.emit(f"  复制到:   {model_dir}\n")
            
            # 调用 C++ 程序
            result = subprocess.run(
                [self.node_path, self.input_config_dir, self.output_configs_dir],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.stdout:
                self.log_signal.emit(result.stdout)
            if result.stderr:
                self.log_signal.emit(result.stderr)
            
            if result.returncode == 0:
                # 复制所有文件到 model 目录
                self.log_signal.emit(f"\n复制配置到 model 目录...")
                copied_count = 0
                for file_path in config_dir.iterdir():
                    if file_path.is_file():
                        shutil.copy2(file_path, model_dir / file_path.name)
                        self.log_signal.emit(f"  复制: {file_path.name}")
                        copied_count += 1
                
                self.log_signal.emit(f"  共复制 {copied_count} 个文件")
                self.finished_signal.emit(True, "Sensing坐标系配置生成成功！已复制到 config 和 model 目录")
            else:
                self.finished_signal.emit(False, f"程序返回错误码: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            self.finished_signal.emit(False, "程序执行超时")
        except Exception as e:
            import traceback
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, f"执行失败: {str(e)}")


class MainWindow(QMainWindow):
    """主窗口"""
    
    # 脚本所在目录
    SCRIPT_DIR = Path(__file__).parent
    DATA_DIR = SCRIPT_DIR / "data"
    
    # 车厂与 raw_config 目录的映射
    OEM_CONFIG_MAP = {
        "OEM-A (3颗Radar)": "raw_config_oem_a",
        "OEM-B (3颗Radar)": "raw_config_oem_b",
        "OEM-C (3颗Radar)": "raw_config_oem_c",
        "OEM-D (5颗Radar)": "raw_config_oem_d",
        "OEM-E (5颗Radar)": "raw_config_oem_e",
    }
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        self.setWindowTitle("车辆配置文件生成工具")
        self.setMinimumSize(1200, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 左右分栏（可拖拽调整大小）
        main_splitter = QSplitter(Qt.Horizontal)
        main_splitter.setHandleWidth(8)
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #888;
            }
        """)
        
        # ========== 左侧：输入控件区 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        
        # 1. 车厂选择
        oem_group = QGroupBox("1. 选择车厂")
        oem_layout = QHBoxLayout(oem_group)
        self.oem_combo = QComboBox()
        self.oem_combo.addItems(list(self.OEM_CONFIG_MAP.keys()))
        self.oem_combo.setCurrentText("OEM-C (3颗Radar)")  # 默认选择
        self.oem_combo.currentTextChanged.connect(self.on_oem_changed)
        oem_layout.addWidget(self.oem_combo)
        left_layout.addWidget(oem_group)
        
        # 2. Raw Config 目录选择（根据车厂自动填充，支持拖拽）
        raw_group = QGroupBox("2. 原始配置目录 (可拖拽)")
        raw_layout = QHBoxLayout(raw_group)
        self.raw_config_edit = DragDropLineEdit()
        self.raw_config_edit.setPlaceholderText("选择车厂后自动填充，或拖拽目录到此处...")
        raw_btn = QPushButton("浏览...")
        raw_btn.clicked.connect(self.select_raw_config)
        raw_layout.addWidget(self.raw_config_edit)
        raw_layout.addWidget(raw_btn)
        left_layout.addWidget(raw_group)
        
        # 3. Excel文件选择（支持拖拽）
        excel_group = QGroupBox("3. 选择Excel文件 (可拖拽)")
        excel_layout = QHBoxLayout(excel_group)
        self.excel_path_edit = DragDropLineEdit()
        self.excel_path_edit.setPlaceholderText("点击浏览或拖拽Excel文件到此处...")
        self.excel_path_edit.fileDropped.connect(self.update_output_paths)  # 拖拽后自动更新输出路径
        excel_btn = QPushButton("浏览...")
        excel_btn.clicked.connect(self.select_excel_file)
        excel_layout.addWidget(self.excel_path_edit)
        excel_layout.addWidget(excel_btn)
        left_layout.addWidget(excel_group)
        
        # 4. 输出目录（自动生成，点击可打开）
        output_group = QGroupBox("4. 输出目录 (自动生成，点击打开)")
        output_layout = QVBoxLayout(output_group)
        
        # 中间输出目录
        mid_label = QLabel("中间输出: calib_config_excel/")
        mid_label.setStyleSheet("color: #888; font-size: 11px;")
        self.output_config_edit = ClickableLineEdit()
        self.output_config_edit.setReadOnly(True)
        self.output_config_edit.setStyleSheet("background-color: #f0f0f0; color: #0066cc; cursor: pointer;")
        self.output_config_edit.setToolTip("点击打开目录")
        self.output_config_edit.clicked.connect(lambda: self.open_directory(self.output_config_edit.text()))
        output_layout.addWidget(mid_label)
        output_layout.addWidget(self.output_config_edit)
        
        # 最终输出目录
        final_label = QLabel("最终输出: calib_config_sensing_coord/config/ 和 model/")
        final_label.setStyleSheet("color: #888; font-size: 11px;")
        self.final_output_edit = ClickableLineEdit()
        self.final_output_edit.setReadOnly(True)
        self.final_output_edit.setStyleSheet("background-color: #f0f0f0; color: #0066cc; cursor: pointer;")
        self.final_output_edit.setToolTip("点击打开目录")
        self.final_output_edit.clicked.connect(lambda: self.open_directory(self.final_output_edit.text()))
        output_layout.addWidget(final_label)
        output_layout.addWidget(self.final_output_edit)
        
        left_layout.addWidget(output_group)
        
        # 主毫米波雷达类型选择
        radar_type_group = QGroupBox("主毫米波雷达类型")
        radar_type_layout = QHBoxLayout(radar_type_group)
        self.radar_type_combo = QComboBox()
        self.radar_type_combo.addItems(["normal_radar", "4d_radar"])
        radar_type_layout.addWidget(self.radar_type_combo)
        left_layout.addWidget(radar_type_group)
        
        # 一键生成按钮
        self.generate_btn = QPushButton("🚀 一键生成配置")
        self.generate_btn.setMinimumHeight(50)
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-size: 15px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #cccccc; }
        """)
        self.generate_btn.clicked.connect(self.generate_all)
        left_layout.addWidget(self.generate_btn)
        
        left_layout.addStretch()
        
        # ========== 右侧：日志 + 3D可视化（可拖拽调整大小） ==========
        right_splitter = QSplitter(Qt.Vertical)
        right_splitter.setHandleWidth(8)
        right_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
                border-radius: 2px;
            }
            QSplitter::handle:hover {
                background-color: #888;
            }
        """)
        
        # 上部：运行日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        log_layout.setContentsMargins(5, 5, 5, 5)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
            }
        """)
        log_layout.addWidget(self.log_text)
        
        # 下部：3D可视化
        viz_group = QGroupBox("3D 传感器布置可视化")
        viz_layout = QVBoxLayout(viz_group)
        viz_layout.setContentsMargins(5, 5, 5, 5)
        self.gl_widget = gl.GLViewWidget()
        self.gl_widget.setCameraPosition(distance=6000, elevation=35, azimuth=135)
        self.gl_widget.setMinimumHeight(200)
        viz_layout.addWidget(self.gl_widget)
        
        # 初始化3D场景
        self.init_3d_scene()
        
        # 显示配置选择指南
        self.show_config_guide()
        
        # 添加到垂直分割器
        right_splitter.addWidget(log_group)
        right_splitter.addWidget(viz_group)
        right_splitter.setSizes([250, 450])  # 初始大小比例
        right_splitter.setCollapsible(0, False)  # 禁止完全折叠
        right_splitter.setCollapsible(1, False)
        
        right_widget = right_splitter
        
        # 添加到主分割器
        main_splitter.addWidget(left_widget)
        main_splitter.addWidget(right_widget)
        main_splitter.setSizes([400, 800])  # 初始左右宽度
        main_splitter.setCollapsible(0, False)
        main_splitter.setCollapsible(1, False)
        
        main_layout.addWidget(main_splitter)
    
    def init_3d_scene(self):
        """初始化3D场景：车辆、轮子、坐标轴"""
        self.visualizer = SensorVisualizer(self.gl_widget)
        self.visualizer.init_scene(front_height=300, body_height=500, rear_height=400)
    
    def show_config_guide(self):
        """显示车厂配置指南"""
        guide = """
┌───────────────────────────────────────────────────────────────────────────────────────┐
│                               【车厂配置选择指南】                                       │
├──────────────────┬────────────────────────────────────────────────────────────────────┤
│ 车厂             │ 传感器配置                                                         │
├──────────────────┼────────────────────────────────────────────────────────────────────┤
│ OEM-A            │ 相机: 11V (平台 A 默认内参)                                        │
│                  │ Lidar: 1颗                                                         │
│                  │ Radar: 3颗 (MRR 通过 Excel 型号判断是否 4D, 补盲为普通)             │
│                  │ USS: 12颗 (默认 vendor_a_ 前缀, Excel 厂商列自动识别)              │
│                  │ IMU: 1个                                                           │
├──────────────────┼────────────────────────────────────────────────────────────────────┤
│ OEM-B            │ 相机: 11V (平台 B 默认内参)                                        │
│                  │ Lidar: 1颗                                                         │
│                  │ Radar: 3颗 (均为非 4D Radar)                                       │
│                  │ USS: 12颗 (默认 vendor_a_ 前缀)                                    │
│                  │ IMU: 1个                                                           │
├──────────────────┼────────────────────────────────────────────────────────────────────┤
│ OEM-C            │ 相机: 11V (平台 C 默认内参)                                        │
│ (3颗Radar)       │ Lidar: 1颗                                                         │
│                  │ Radar: 3颗 (主 radar 自身坐标系, 补盲 radar 车辆坐标系)             │
│                  │ USS: 12颗 (默认 vendor_b_ 前缀)                                    │
│                  │ IMU: 1个                                                           │
├──────────────────┼────────────────────────────────────────────────────────────────────┤
│ OEM-D            │ 相机: 11V (平台 D 默认内参)                                        │
│ (5颗Radar)       │ Lidar: 1颗                                                         │
│                  │ Radar: 5颗 (主 radar 自身坐标系, 补盲 radar 车辆坐标系)             │
│                  │ USS: 12颗 (默认 vendor_b_ 前缀)                                    │
│                  │ IMU: 1个                                                           │
├──────────────────┼────────────────────────────────────────────────────────────────────┤
│ OEM-E            │ 相机: 11V (平台 E 默认内参)                                        │
│ (5颗Radar)       │ Lidar: 1颗                                                         │
│                  │ Radar: 5颗 (主 radar 车辆坐标系)                                   │
│                  │ USS: 12颗 (待补充)                                                 │
│                  │ IMU: 1个                                                           │
└──────────────────┴────────────────────────────────────────────────────────────────────┘

📌 请根据实际车型选择对应的车厂配置！
"""
        self.log_text.setText(guide)
    
    def update_sensors_from_data(self, sensor_data: dict):
        """根据Excel传感器数据更新3D视图"""
        count = self.visualizer.update_from_data(sensor_data)
        self.log_text.append(f"\n🎨 3D视图已更新 {count} 个传感器")
    
    def on_oem_changed(self, oem_name):
        """车厂选择变化时，自动更新 raw_config 目录"""
        config_dir_name = self.OEM_CONFIG_MAP.get(oem_name, "raw_config")
        raw_config_path = self.DATA_DIR / config_dir_name
        self.raw_config_edit.setText(str(raw_config_path))
    
    def select_excel_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择Excel文件", "",
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        if file_path:
            self.excel_path_edit.setText(file_path)
            self.update_output_paths(file_path)
    
    def update_output_paths(self, excel_path):
        """根据 Excel 路径自动更新输出目录"""
        if excel_path:
            excel_dir = Path(excel_path).parent
            # 中间输出目录
            mid_output = excel_dir / "calib_config_excel"
            self.output_config_edit.setText(str(mid_output))
            # 最终输出目录 (config)
            final_output = excel_dir / "calib_config_sensing_coord" / "config"
            self.final_output_edit.setText(str(final_output))
    
    def select_raw_config(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择原始配置目录", "")
        if dir_path:
            self.raw_config_edit.setText(dir_path)
    
    def open_directory(self, path):
        """打开目录（使用系统文件管理器）"""
        if not path:
            return
        
        dir_path = Path(path)
        # 如果目录不存在，尝试打开父目录
        if not dir_path.exists():
            dir_path = dir_path.parent
            if not dir_path.exists():
                QMessageBox.warning(self, "提示", f"目录不存在，请先执行生成操作:\n{path}")
                return
        
        import platform
        import subprocess
        
        system = platform.system()
        try:
            if system == "Linux":
                subprocess.Popen(["xdg-open", str(dir_path)])
            elif system == "Darwin":  # macOS
                subprocess.Popen(["open", str(dir_path)])
            elif system == "Windows":
                subprocess.Popen(["explorer", str(dir_path)])
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法打开目录: {e}")
    
    def generate_all(self):
        """一键生成配置（Excel转CFG + Sensing坐标系配置）"""
        excel_path = self.excel_path_edit.text()
        raw_config = self.raw_config_edit.text()
        output_config = self.output_config_edit.text()
        
        if not excel_path:
            QMessageBox.warning(self, "警告", "请选择Excel文件")
            return
        if not raw_config:
            QMessageBox.warning(self, "警告", "请选择原始配置目录")
            return
        if not output_config:
            QMessageBox.warning(self, "警告", "请选择中间输出目录")
            return
        
        if not Path(excel_path).exists():
            QMessageBox.warning(self, "警告", f"Excel文件不存在: {excel_path}")
            return
        if not Path(raw_config).exists():
            QMessageBox.warning(self, "警告", f"原始配置目录不存在: {raw_config}")
            return
        
        main_radar_type = self.radar_type_combo.currentText()
        
        try:
            radar_properties = VehicleConfigUpdater.select_radar_properties(
                main_radar_type, raw_config
            )
        except ValueError as e:
            QMessageBox.warning(self, "警告", str(e))
            return
        
        # 检查 Excel 是否有传感器型号列
        has_model_col = VehicleConfigUpdater.check_excel_has_model_column(excel_path)
        if not has_model_col:
            reply = QMessageBox.warning(
                self, 
                "⚠ 警告", 
                "此 Excel 没有传感器型号列！\n\n"
                "需要人工确认 raw_config 中的 USS frame_id 是否正确。\n\n"
                "是否继续转换？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        oem_name = self.oem_combo.currentText()
        
        self.log_text.clear()
        self.log_text.append("=" * 60)
        self.log_text.append("【当前配置】请确认以下信息是否正确：")
        self.log_text.append("=" * 60)
        self.log_text.append(f"  车厂:         {oem_name}")
        self.log_text.append(f"  Excel文件:    {excel_path}")
        self.log_text.append(f"  原始配置目录: {raw_config}")
        self.log_text.append(f"  中间输出:     {output_config}")
        self.log_text.append(f"  最终输出:     {self.final_output_edit.text()}")
        self.log_text.append(f"  主雷达类型:   {main_radar_type}")
        self.log_text.append("=" * 60)
        self.log_text.append("\n【步骤1】开始 Excel 转 CFG...\n")
        
        self.generate_btn.setEnabled(False)
        self.generate_btn.setText("生成中 (1/2)...")
        
        self.worker = WorkerThread(excel_path, raw_config, output_config, radar_properties)
        self.worker.log_signal.connect(self.append_log)
        self.worker.sensor_data_signal.connect(self.update_sensors_from_data)
        self.worker.warning_signal.connect(self.show_warning)
        self.worker.finished_signal.connect(self.on_step1_finished)
        self.worker.start()
    
    def append_log(self, text):
        self.log_text.append(text)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def show_warning(self, message):
        """显示警告弹窗"""
        QMessageBox.warning(self, "⚠ 警告", message)
    
    def on_step1_finished(self, success, message):
        """步骤1完成（Excel转CFG），自动启动步骤2"""
        if success:
            self.log_text.append(f"\n✅ {message}")
            self.log_text.append("\n【步骤2】开始生成 Sensing 坐标系配置...\n")
            
            self.generate_btn.setText("生成中 (2/2)...")
            
            # 自动执行步骤2
            input_config_dir = self.output_config_edit.text()
            output_configs_dir = self.final_output_edit.text()
            
            self.sensing_worker = SensingWorkerThread(input_config_dir, output_configs_dir)
            self.sensing_worker.log_signal.connect(self.append_log)
            self.sensing_worker.finished_signal.connect(self.on_step2_finished)
            self.sensing_worker.start()
        else:
            self.generate_btn.setEnabled(True)
            self.generate_btn.setText("🚀 一键生成配置")
            self.log_text.append(f"\n❌ {message}")
            QMessageBox.critical(self, "错误", message)
    
    def on_step2_finished(self, success, message):
        """步骤2完成（生成Sensing坐标系配置），全部完成"""
        self.generate_btn.setEnabled(True)
        self.generate_btn.setText("🚀 一键生成配置")
        
        if success:
            self.log_text.append(f"\n✅ {message}")
            self.log_text.append("\n" + "=" * 60)
            self.log_text.append("🎉 全部生成完成！")
            self.log_text.append("=" * 60)
            QMessageBox.information(self, "成功", "配置文件全部生成完成！\n\n"
                                   f"中间输出: {self.output_config_edit.text()}\n"
                                   f"最终输出: {self.final_output_edit.text()}")
        else:
            self.log_text.append(f"\n❌ {message}")
            QMessageBox.critical(self, "错误", message)


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
