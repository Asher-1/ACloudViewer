#!/usr/bin/env python3
"""
ScaleBar字体大小优化测试脚本
测试ScaleBar在不同平台和分辨率下的字体大小优化效果
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

def simulate_scalebar_font_optimization():
    """模拟ScaleBar字体大小优化"""
    print("=" * 60)
    print("ScaleBar字体大小优化测试")
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
    
    # 模拟不同DPI缩放值
    dpi_scales = [1.0, 1.5, 2.0, 2.5]
    base_font_size = 18  # ScaleBar的基础字体大小
    
    print("ScaleBar字体大小优化测试:")
    print("-" * 50)
    
    for dpi_scale in dpi_scales:
        print(f"DPI缩放: {dpi_scale}")
        
        # 模拟原始字体大小（未优化）
        original_font_size = int(base_font_size * dpi_scale)
        
        # 模拟优化后的字体大小
        if system_info['platform'] == "Darwin":  # macOS
            if dpi_scale > 1:
                # Retina显示器：使用较小的字体避免过度放大
                optimized_font_size = max(12, base_font_size - (dpi_scale - 1) * 3)
            else:
                optimized_font_size = base_font_size
        elif system_info['platform'] == "Windows":
            # 假设高DPI显示器
            if dpi_scale > 1.2:
                optimized_font_size = max(12, base_font_size - 2)
            elif dpi_scale < 0.96:
                optimized_font_size = base_font_size + 2
            else:
                optimized_font_size = base_font_size
        elif system_info['platform'] == "Linux":
            # 假设高分辨率显示器
            optimized_font_size = max(12, base_font_size - 2)
        else:
            optimized_font_size = base_font_size
        
        # 确保在合理范围内
        optimized_font_size = int(max(10, min(32, optimized_font_size)))
        
        # 模拟平台感知的DPI缩放
        if system_info['platform'] == "Darwin":  # macOS
            if dpi_scale > 1:
                adjusted_scale = 1.0 + (dpi_scale - 1.0) * 0.6
            else:
                adjusted_scale = dpi_scale
        elif system_info['platform'] == "Windows":
            if dpi_scale > 1.2:
                adjusted_scale = min(dpi_scale, 1.4)
            elif dpi_scale < 0.96:
                adjusted_scale = max(dpi_scale, 1.0)
            else:
                adjusted_scale = dpi_scale
        elif system_info['platform'] == "Linux":
            if dpi_scale > 1.5:
                adjusted_scale = min(dpi_scale, 1.2)
            else:
                adjusted_scale = dpi_scale
        else:
            adjusted_scale = dpi_scale
        
        # 确保缩放在合理范围内
        adjusted_scale = max(0.8, min(1.8, adjusted_scale))
        
        print(f"  原始字体大小: {original_font_size:2d}")
        print(f"  优化后字体大小: {optimized_font_size:2d}")
        print(f"  原始DPI缩放: {dpi_scale:4.1f}")
        print(f"  调整后DPI缩放: {adjusted_scale:4.1f}")
        print(f"  优化效果: {original_font_size - optimized_font_size:+2d} 像素")
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
    
    # 检查是否包含新的优化函数
    try:
        with open("libs/PCLEngine/VTKExtensions/Widgets/ScaleBar.h", 'r', encoding='utf-8') as f:
            content = f.read()
            if "getOptimizedFontSize" in content:
                print("✓ getOptimizedFontSize 函数已添加")
            else:
                print("✗ getOptimizedFontSize 函数未找到")
            
            if "getPlatformAwareDPIScale" in content:
                print("✓ getPlatformAwareDPIScale 函数已添加")
            else:
                print("✗ getPlatformAwareDPIScale 函数未找到")
    except (OSError, UnicodeDecodeError) as e:
        print(f"✗ 无法读取文件: {e}")
    
    try:
        with open("libs/PCLEngine/VTKExtensions/Widgets/ScaleBar.cpp", 'r', encoding='utf-8') as f:
            content = f.read()
            if "getOptimizedFontSize(18)" in content:
                print("✓ 构造函数中使用了优化的字体大小")
            else:
                print("✗ 构造函数中未使用优化的字体大小")
            
            if "getPlatformAwareDPIScale()" in content:
                print("✓ 使用了平台感知的DPI缩放")
            else:
                print("✗ 未使用平台感知的DPI缩放")
    except (OSError, UnicodeDecodeError) as e:
        print(f"✗ 无法读取文件: {e}")

if __name__ == "__main__":
    print("ScaleBar字体大小优化测试工具")
    print()
    
    # 检查文件状态
    check_scalebar_files()
    print()
    
    # 运行字体优化测试
    simulate_scalebar_font_optimization() 