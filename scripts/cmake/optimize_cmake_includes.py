#!/usr/bin/env python3
"""
优化CMake文件，移除不必要的include并使用全局变量
"""

import sys
from pathlib import Path

def optimize_file(file_path):
    """优化单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.split('\n')
        optimized_lines = []
        removed_includes = []
        
        for line in lines:
            # 检查是否是include cmake/CMakeVersionConfig.cmake
            if 'include(cmake/CMakeVersionConfig.cmake)' in line.strip():
                removed_includes.append(line.strip())
                continue
            
            # 检查是否是set_xxx_cmake_minimum_required()调用
            if any(func in line.strip() for func in [
                'set_subproject_cmake_minimum_required()',
                'set_plugin_cmake_minimum_required()',
                'set_thirdparty_cmake_minimum_required()'
            ]):
                removed_includes.append(line.strip())
                continue
            
            # 检查cmake_minimum_required是否使用硬编码版本
            if 'cmake_minimum_required(VERSION' in line:
                # 确定项目类型
                project_type = determine_project_type(file_path)
                
                if project_type == 'plugin':
                    new_line = line.replace('VERSION 3.10', 'VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION}')
                    new_line = new_line.replace('VERSION 3.15', 'VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION}')
                    new_line = new_line.replace('VERSION 3.18', 'VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION}')
                elif project_type == 'thirdparty':
                    new_line = line.replace('VERSION 3.10', 'VERSION ${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION}')
                    new_line = new_line.replace('VERSION 3.15', 'VERSION ${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION}')
                    new_line = new_line.replace('VERSION 3.18', 'VERSION ${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION}')
                else:  # subproject
                    new_line = line.replace('VERSION 3.10', 'VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION}')
                    new_line = new_line.replace('VERSION 3.15', 'VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION}')
                    new_line = new_line.replace('VERSION 3.18', 'VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION}')
                
                optimized_lines.append(new_line)
                continue
            
            optimized_lines.append(line)
        
        # 如果有优化，写回文件
        if removed_includes:
            new_content = '\n'.join(optimized_lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return removed_includes
        
        return []
    except (IOError, OSError) as e:
        print(f"Error optimizing {file_path}: {e}")
        return []

def determine_project_type(file_path):
    """根据文件路径确定项目类型"""
    path_str = str(file_path)
    
    if 'plugins/' in path_str:
        return 'plugin'
    if '3rdparty/' in path_str or 'extern/' in path_str:
        return 'thirdparty'
    if 'libs/' in path_str or 'core/' in path_str:
        return 'subproject'
    return 'subproject'

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python optimize_cmake_includes.py <action>")
        print("Actions:")
        print("  check    - Check which files can be optimized")
        print("  optimize - Optimize files by removing unnecessary includes")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "check":
        print("Checking files that can be optimized...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        optimizable_count = 0
        
        for file_path in cmake_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有不必要的include或函数调用
                if ('include(cmake/CMakeVersionConfig.cmake)' in content or
                    'set_subproject_cmake_minimum_required()' in content or
                    'set_plugin_cmake_minimum_required()' in content or
                    'set_thirdparty_cmake_minimum_required()' in content):
                    project_type = determine_project_type(file_path)
                    print(f"Optimizable ({project_type}): {file_path}")
                    optimizable_count += 1
            except (IOError, OSError) as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Found {optimizable_count} files that can be optimized")
        
    elif action == "optimize":
        print("Optimizing files...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        optimized_count = 0
        
        for file_path in cmake_files:
            removed_items = optimize_file(str(file_path))
            if removed_items:
                project_type = determine_project_type(file_path)
                print(f"Optimized ({project_type}): {file_path}")
                print(f"  Removed: {', '.join(removed_items)}")
                optimized_count += 1
        
        print(f"Optimized {optimized_count} files")
        
    else:
        print("Invalid action")
        sys.exit(1)

if __name__ == "__main__":
    main() 