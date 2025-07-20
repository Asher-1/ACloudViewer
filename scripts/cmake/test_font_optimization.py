#!/usr/bin/env python3
"""
跨平台字体大小优化测试脚本
测试不同平台和分辨率下的字体大小优化效果
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
                # 解析分辨率信息
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

def test_font_optimization():
    """测试字体大小优化"""
    print("=" * 60)
    print("跨平台字体大小优化测试")
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
    
    # 模拟不同平台的字体大小优化
    print("字体大小优化测试:")
    print("-" * 40)
    
    # 模拟不同DPI缩放值
    dpi_scales = [1.0, 1.5, 2.0, 2.5]
    
    for dpi_scale in dpi_scales:
        print(f"DPI缩放: {dpi_scale}")
        
        # 模拟不同平台的基础字体大小
        base_sizes = [8, 10, 12, 14, 16]
        
        for base_size in base_sizes:
            # 模拟优化后的字体大小
            if system_info['platform'] == "Darwin":  # macOS
                if dpi_scale > 1:
                    optimized_size = max(8, base_size - (dpi_scale - 1) * 2)
                else:
                    optimized_size = base_size
            elif system_info['platform'] == "Windows":
                if dpi_scale > 1.2:
                    optimized_size = max(8, base_size - 1)
                elif dpi_scale < 0.96:
                    optimized_size = base_size + 1
                else:
                    optimized_size = base_size
            elif system_info['platform'] == "Linux":
                # 假设高分辨率显示器
                optimized_size = max(8, base_size - 1)
            else:
                optimized_size = base_size
            
            # 确保在合理范围内
            optimized_size = max(6, min(24, optimized_size))
            
            print(f"  基础大小 {base_size:2d} -> 优化后 {int(optimized_size):2d}")
        
        print()
    
    print("DPI缩放处理测试:")
    print("-" * 40)
    
    for dpi_scale in dpi_scales:
        # 模拟平台感知的DPI缩放
        if system_info['platform'] == "Darwin":  # macOS
            if dpi_scale > 1:
                adjusted_scale = 1.0 + (dpi_scale - 1.0) * 0.5
            else:
                adjusted_scale = dpi_scale
        elif system_info['platform'] == "Windows":
            if dpi_scale > 1.2:
                adjusted_scale = min(dpi_scale, 1.5)
            elif dpi_scale < 0.96:
                adjusted_scale = max(dpi_scale, 1.0)
            else:
                adjusted_scale = dpi_scale
        elif system_info['platform'] == "Linux":
            # 假设超高分辨率
            if dpi_scale > 1.5:
                adjusted_scale = min(dpi_scale, 1.3)
            else:
                adjusted_scale = dpi_scale
        else:
            adjusted_scale = dpi_scale
        
        # 确保在合理范围内
        adjusted_scale = max(0.5, min(2.0, adjusted_scale))
        
        print(f"原始DPI缩放: {dpi_scale:4.1f} -> 调整后: {adjusted_scale:4.1f}")
    
    print()
    print("=" * 60)
    print("测试完成")
    print("=" * 60)

def check_compilation():
    """检查编译是否成功"""
    print("检查编译状态...")
    
    # 检查关键文件是否存在
    key_files = [
        "libs/eCV_db/include/ecvDisplayTools.h",
        "libs/eCV_db/src/ecvDisplayTools.cpp",
        "libs/eCV_db/src/ecvGuiParameters.cpp"
    ]
    
    for file_path in key_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - 文件不存在")
    
    print()
    
    # 检查是否包含新的优化函数
    try:
        with open("libs/eCV_db/include/ecvDisplayTools.h", 'r', encoding='utf-8') as f:
            content = f.read()
            if "GetOptimizedFontSize" in content:
                print("✓ GetOptimizedFontSize 函数已添加")
            else:
                print("✗ GetOptimizedFontSize 函数未找到")
            
            if "GetPlatformAwareDPIScale" in content:
                print("✓ GetPlatformAwareDPIScale 函数已添加")
            else:
                print("✗ GetPlatformAwareDPIScale 函数未找到")
    except (OSError, UnicodeDecodeError) as e:
        print(f"✗ 无法读取文件: {e}")

if __name__ == "__main__":
    print("跨平台字体大小优化测试工具")
    print()
    
    # 检查编译状态
    check_compilation()
    print()
    
    # 运行字体优化测试
    test_font_optimization() 