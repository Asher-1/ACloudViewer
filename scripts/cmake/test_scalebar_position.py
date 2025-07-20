#!/usr/bin/env python3
"""
ScaleBar位置调整测试脚本
测试ScaleBar在窗口底部居中的位置调整效果
"""

import os
import platform
import subprocess

def get_system_info():
    """获取系统信息"""
    info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'python_version': platform.python_version()
    }
    
    # 获取屏幕分辨率（如果可能）
    try:
        if platform.system() == "Darwin":  # macOS
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'Resolution:' in line:
                        info['resolution'] = line.strip()
                        break
        elif platform.system() == "Windows":
            result = subprocess.run(['wmic', 'desktopmonitor', 'get', 'ScreenWidth,ScreenHeight'], 
                                  capture_output=True, text=True, check=False)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:
                    info['resolution'] = lines[1].strip()
        elif platform.system() == "Linux":
            result = subprocess.run(['xrandr'], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if '*' in line and 'x' in line:
                        info['resolution'] = line.split()[0]
                        break
    except (subprocess.SubprocessError, OSError) as e:
        info['resolution'] = f"无法获取: {e}"
    
    return info

def simulate_scalebar_positioning():
    """模拟ScaleBar位置调整"""
    print("=" * 60)
    print("ScaleBar位置调整测试")
    print("=" * 60)
    
    # 获取系统信息
    system_info = get_system_info()
    print(f"操作系统: {system_info['platform']}")
    print(f"系统版本: {system_info['platform_version']}")
    print(f"架构: {system_info['architecture']}")
    print(f"处理器: {system_info['processor']}")
    print(f"分辨率: {system_info.get('resolution', '未知')}")
    print(f"Python版本: {system_info['python_version']}")
    print()
    
    # 模拟不同窗口大小
    window_sizes = [
        (800, 600),   # 小窗口
        (1024, 768),  # 中等窗口
        (1920, 1080), # 1080p
        (2560, 1440), # 2K
        (3840, 2160)  # 4K
    ]
    
    dpi_scales = [1.0, 1.5, 2.0]
    
    print("ScaleBar位置计算测试:")
    print("-" * 50)
    
    for win_width, win_height in window_sizes:
        print(f"窗口大小: {win_width} x {win_height}")
        
        for dpi_scale in dpi_scales:
            # 计算ScaleBar参数
            bar_pixel_len = int((win_width / 6.0) * dpi_scale)  # 约为窗口宽度的1/6
            bottom_margin = 50.0 * dpi_scale  # 底部边距
            center_x = win_width / 2.0  # 窗口中心X坐标
            bottom_y = bottom_margin  # 底部Y坐标
            
            # 计算ScaleBar的起始和结束位置（居中）
            p1_x = center_x - bar_pixel_len / 2.0  # 线条左端点X
            p1_y = bottom_y  # 线条左端点Y
            p2_x = center_x + bar_pixel_len / 2.0  # 线条右端点X
            p2_y = bottom_y  # 线条右端点Y
            
            # 计算文本位置
            text_x = center_x  # 文本X坐标（居中）
            text_y = bottom_y - 20.0 * dpi_scale  # 文本Y坐标（线条下方）
            
            print(f"  DPI缩放: {dpi_scale}")
            print(f"    比例尺长度: {bar_pixel_len} 像素")
            print(f"    底部边距: {bottom_margin:.1f} 像素")
            print(f"    线条位置: ({p1_x:.1f}, {p1_y:.1f}) -> ({p2_x:.1f}, {p2_y:.1f})")
            print(f"    文本位置: ({text_x:.1f}, {text_y:.1f})")
            print(f"    居中偏移: {abs(center_x - (p1_x + p2_x) / 2):.1f} 像素")
            print()
    
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

def check_scalebar_files():
    """检查ScaleBar相关文件"""
    print("检查ScaleBar文件状态...")
    
    # 检查关键文件是否存在
    key_files = [
        "libs/PCLEngine/VTKExtensions/Widgets/ScaleBar.h",
        "libs/PCLEngine/VTKExtensions/Widgets/ScaleBar.cpp"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - 文件不存在")
    
    print()
    
    # 检查是否包含位置调整的代码
    try:
        with open("libs/PCLEngine/VTKExtensions/Widgets/ScaleBar.cpp", 'r', encoding='utf-8') as f:
            content = f.read()
            if "centerX" in content:
                print("✓ 使用了窗口中心X坐标计算")
            else:
                print("✗ 未使用窗口中心X坐标计算")
            
            if "bottomY" in content:
                print("✓ 使用了底部Y坐标计算")
            else:
                print("✗ 未使用底部Y坐标计算")
            
            if "bottomMargin" in content:
                print("✓ 使用了底部边距设置")
            else:
                print("✗ 未使用底部边距设置")
            
            if "SetPosition" in content:
                print("✓ 包含位置设置代码")
            else:
                print("✗ 未包含位置设置代码")
    except (OSError, UnicodeDecodeError) as e:
        print(f"✗ 无法读取文件: {e}")

if __name__ == "__main__":
    print("ScaleBar位置调整测试工具")
    print()
    
    # 检查文件状态
    check_scalebar_files()
    print()
    
    # 运行位置调整测试
    simulate_scalebar_positioning() 