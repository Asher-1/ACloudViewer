#!/usr/bin/env python3
"""
修复cmake_minimum_required的位置问题，确保它在文件开头
"""

import sys
from pathlib import Path

def fix_cmake_minimum_required_position(file_path):
    """修复单个文件中cmake_minimum_required的位置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否已经有cmake_minimum_required在开头
        lines = content.split('\n')
        first_non_empty_line = None
        for i, line in enumerate(lines):
            if line.strip() and not line.strip().startswith('#'):
                first_non_empty_line = i
                break
        
        if first_non_empty_line is None:
            return False
        
        # 检查第一行是否已经是cmake_minimum_required
        if 'cmake_minimum_required' in lines[first_non_empty_line]:
            return False
        
        # 查找cmake_minimum_required的位置
        cmake_min_line = None
        for i, line in enumerate(lines):
            if 'cmake_minimum_required' in line:
                cmake_min_line = i
                break
        
        if cmake_min_line is None:
            return False
        
        # 如果cmake_minimum_required不在开头，需要移动它
        if cmake_min_line > first_non_empty_line:
            # 提取cmake_minimum_required行
            cmake_min_content = lines[cmake_min_line]
            
            # 删除原来的行
            lines.pop(cmake_min_line)
            
            # 在开头插入（在注释之后）
            insert_position = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('#'):
                    insert_position = i + 1
                else:
                    break
            
            lines.insert(insert_position, cmake_min_content)
            lines.insert(insert_position + 1, '')  # 添加空行
            
            # 写回文件
            new_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
    except (IOError, OSError) as e:
        print(f"Error fixing {file_path}: {e}")
        return False

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python fix_cmake_minimum_required_position.py <action>")
        print("Actions:")
        print("  check    - Check which files have cmake_minimum_required in wrong position")
        print("  fix      - Fix cmake_minimum_required position in files")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "check":
        print("Checking files with cmake_minimum_required in wrong position...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        problematic_count = 0
        
        for file_path in cmake_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = content.split('\n')
                first_non_empty_line = None
                for i, line in enumerate(lines):
                    if line.strip() and not line.strip().startswith('#'):
                        first_non_empty_line = i
                        break
                
                if first_non_empty_line is not None:
                    # 检查第一行是否不是cmake_minimum_required
                    if ('cmake_minimum_required' in content and 
                        'cmake_minimum_required' not in lines[first_non_empty_line]):
                        print(f"Problematic: {file_path}")
                        problematic_count += 1
            except (IOError, OSError) as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Found {problematic_count} files with cmake_minimum_required in wrong position")
        
    elif action == "fix":
        print("Fixing cmake_minimum_required position...")
        
        # 查找所有CMakeLists.txt文件
        cmake_files = list(Path('.').rglob('CMakeLists.txt'))
        fixed_count = 0
        
        for file_path in cmake_files:
            if fix_cmake_minimum_required_position(str(file_path)):
                print(f"Fixed: {file_path}")
                fixed_count += 1
        
        print(f"Fixed {fixed_count} files")
        
    else:
        print("Invalid action")
        sys.exit(1)

if __name__ == "__main__":
    main() 