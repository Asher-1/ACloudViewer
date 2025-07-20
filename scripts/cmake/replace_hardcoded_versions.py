#!/usr/bin/env python3
"""
替换硬编码的CMake版本号为变量
"""

import sys
from pathlib import Path

def replace_hardcoded_versions(file_path):
    """替换单个文件中的硬编码版本号"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 确定项目类型
        project_type = determine_project_type(file_path)
        
        # 根据项目类型选择变量
        if project_type == 'plugin':
            variable = '${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION}'
        elif project_type == 'thirdparty':
            variable = '${CLOUDVIEWER_THIRDPARTY_CMAKE_MINIMUM_VERSION}'
        else:  # subproject
            variable = '${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION}'
        
        # 替换硬编码版本号
        original_content = content
        content = content.replace('VERSION 3.10', f'VERSION {variable}')
        content = content.replace('VERSION 3.15', f'VERSION {variable}')
        content = content.replace('VERSION 3.18', f'VERSION {variable}')
        content = content.replace('VERSION 3.19', f'VERSION {variable}')
        
        # 如果有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    except (IOError, OSError) as e:
        print(f"Error processing {file_path}: {e}")
        return False

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
        print("Usage: python replace_hardcoded_versions.py <action>")
        print("Actions:")
        print("  check    - Check which files have hardcoded versions")
        print("  replace  - Replace hardcoded versions with variables")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "check":
        print("Checking files with hardcoded versions...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        hardcoded_count = 0
        
        for file_path in cmake_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有硬编码版本号
                if ('cmake_minimum_required(VERSION 3.10)' in content or
                    'cmake_minimum_required(VERSION 3.15)' in content or
                    'cmake_minimum_required(VERSION 3.18)' in content or
                    'cmake_minimum_required(VERSION 3.19)' in content):
                    project_type = determine_project_type(file_path)
                    print(f"Hardcoded ({project_type}): {file_path}")
                    hardcoded_count += 1
            except (IOError, OSError) as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Found {hardcoded_count} files with hardcoded versions")
        
    elif action == "replace":
        print("Replacing hardcoded versions...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        replaced_count = 0
        
        for file_path in cmake_files:
            if replace_hardcoded_versions(str(file_path)):
                project_type = determine_project_type(file_path)
                print(f"Replaced ({project_type}): {file_path}")
                replaced_count += 1
        
        print(f"Replaced versions in {replaced_count} files")
        
    else:
        print("Invalid action")
        sys.exit(1)

if __name__ == "__main__":
    main() 