#!/usr/bin/env python3
"""
批量更新CMakeLists.txt文件中的cmake_minimum_required版本
"""

import re
import sys
from pathlib import Path

def update_cmake_version(file_path, new_version):
    """更新单个文件中的cmake_minimum_required版本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换cmake_minimum_required
        pattern = r'cmake_minimum_required\s*\(\s*VERSION\s+[0-9.]+(?:\s*\.\.[0-9.]+)?\s*\)'
        replacement = f'cmake_minimum_required(VERSION {new_version})'
        
        new_content = re.sub(pattern, replacement, content)
        
        if new_content != content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except (IOError, OSError) as e:
        print(f"Error updating {file_path}: {e}")
        return False

def add_cmake_minimum_required(file_path, version):
    """在文件开头添加cmake_minimum_required"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有cmake_minimum_required
        if 'cmake_minimum_required' in content:
            return False
        
        # 检查是否有project声明
        if 'project(' in content:
            new_content = f'cmake_minimum_required(VERSION {version})\n\n{content}'
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except (IOError, OSError) as e:
        print(f"Error adding cmake_minimum_required to {file_path}: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python update_cmake_versions.py <action> [version]")
        print("Actions:")
        print("  update <version>  - Update existing cmake_minimum_required to specified version")
        print("  add <version>     - Add cmake_minimum_required to files that don't have it")
        print("  check             - Check which files need cmake_minimum_required")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "update" and len(sys.argv) >= 3:
        version = sys.argv[2]
        print(f"Updating cmake_minimum_required to version {version}")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        updated_count = 0
        
        for file_path in cmake_files:
            if update_cmake_version(str(file_path), version):
                print(f"Updated: {file_path}")
                updated_count += 1
        
        print(f"Updated {updated_count} files")
        
    elif action == "add" and len(sys.argv) >= 3:
        version = sys.argv[2]
        print(f"Adding cmake_minimum_required version {version} to files that need it")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        added_count = 0
        
        for file_path in cmake_files:
            if add_cmake_minimum_required(str(file_path), version):
                print(f"Added to: {file_path}")
                added_count += 1
        
        print(f"Added cmake_minimum_required to {added_count} files")
        
    elif action == "check":
        print("Checking files that need cmake_minimum_required...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        missing_count = 0
        
        for file_path in cmake_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'project(' in content and 'cmake_minimum_required' not in content:
                    print(f"Missing: {file_path}")
                    missing_count += 1
            except (IOError, OSError) as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Found {missing_count} files missing cmake_minimum_required")
        
    else:
        print("Invalid action or missing version")
        sys.exit(1)

if __name__ == "__main__":
    main() 