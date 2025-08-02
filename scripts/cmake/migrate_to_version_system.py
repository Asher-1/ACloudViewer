#!/usr/bin/env python3
"""
将硬编码的cmake_minimum_required迁移到新的版本管理系统
"""

import re
import sys
from pathlib import Path

def migrate_file(file_path):
    """迁移单个文件到新的版本管理系统"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有include语句
        if 'include(cmake/CMakeVersionConfig.cmake)' in content:
            return False
        
        # 查找cmake_minimum_required行
        pattern = r'cmake_minimum_required\s*\(\s*VERSION\s+[0-9.]+(?:\s*\.\.[0-9.]+)?\s*\)'
        match = re.search(pattern, content)
        
        if match:
            # 确定项目类型
            project_type = determine_project_type(file_path)
            
            # 替换cmake_minimum_required
            if project_type == 'plugin':
                replacement = '''include(cmake/CMakeVersionConfig.cmake)
set_plugin_cmake_minimum_required()'''
            elif project_type == 'subproject':
                replacement = '''include(cmake/CMakeVersionConfig.cmake)
set_subproject_cmake_minimum_required()'''
            elif project_type == 'thirdparty':
                replacement = '''include(cmake/CMakeVersionConfig.cmake)
set_thirdparty_cmake_minimum_required()'''
            else:
                replacement = '''include(cmake/CMakeVersionConfig.cmake)
set_subproject_cmake_minimum_required()'''
            
            new_content = re.sub(pattern, replacement, content)
            
            if new_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)
                return True
        
        return False
    except (IOError, OSError) as e:
        print(f"Error migrating {file_path}: {e}")
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
        print("Usage: python migrate_to_version_system.py <action>")
        print("Actions:")
        print("  check    - Check which files can be migrated")
        print("  migrate  - Migrate files to new version system")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "check":
        print("Checking files that can be migrated to new version system...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        migratable_count = 0
        
        for file_path in cmake_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有硬编码的cmake_minimum_required但没有include
                if ('cmake_minimum_required' in content and 
                    'include(cmake/CMakeVersionConfig.cmake)' not in content):
                    project_type = determine_project_type(file_path)
                    print(f"Migratable ({project_type}): {file_path}")
                    migratable_count += 1
            except (IOError, OSError) as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Found {migratable_count} files that can be migrated")
        
    elif action == "migrate":
        print("Migrating files to new version system...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        migrated_count = 0
        
        for file_path in cmake_files:
            if migrate_file(str(file_path)):
                project_type = determine_project_type(file_path)
                print(f"Migrated ({project_type}): {file_path}")
                migrated_count += 1
        
        print(f"Migrated {migrated_count} files")
        
    else:
        print("Invalid action")
        sys.exit(1)

if __name__ == "__main__":
    main() 