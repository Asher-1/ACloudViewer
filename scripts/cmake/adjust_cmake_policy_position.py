#!/usr/bin/env python3
"""
调整CMAKE_POLICY_VERSION_MINIMUM参数的位置到CMAKE_ARGS的第一行
"""

import sys
import re
from pathlib import Path

def adjust_cmake_policy_position(file_path):
    """调整单个文件中CMAKE_POLICY_VERSION_MINIMUM参数的位置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 查找ExternalProject_Add调用
        pattern = r'ExternalProject_Add\s*\(\s*([^)]+?)\s*\)'
        
        def replace_externalproject(match):
            params = match.group(1)
            
            # 检查是否有CMAKE_ARGS和CMAKE_POLICY_VERSION_MINIMUM
            if 'CMAKE_ARGS' in params and 'CMAKE_POLICY_VERSION_MINIMUM' in params:
                # 找到CMAKE_ARGS的位置
                cmake_args_pattern = r'CMAKE_ARGS\s+([^)]+?)(?=\s+[A-Z_]+:|$)'
                cmake_args_match = re.search(cmake_args_pattern, params, re.DOTALL)
                if cmake_args_match:
                    cmake_args = cmake_args_match.group(1)
                    
                    # 移除现有的CMAKE_POLICY_VERSION_MINIMUM参数
                    cmake_args_cleaned = re.sub(r'\s*-DCMAKE_POLICY_VERSION_MINIMUM=3\.5\s*', '', cmake_args)
                    
                    # 将版本参数放在第一行
                    indent = ''
                    lines = cmake_args_cleaned.split('\n')
                    for line in lines:
                        if line.strip():
                            indent = line[:len(line) - len(line.lstrip())]
                            break
                    
                    new_cmake_args = f'\n{indent}-DCMAKE_POLICY_VERSION_MINIMUM=3.5\n' + cmake_args_cleaned
                    new_params = params.replace(cmake_args, new_cmake_args)
                    return f'ExternalProject_Add({new_params})'
            
            return match.group(0)  # 没有需要调整的，返回原内容
        
        # 执行替换
        new_content = re.sub(pattern, replace_externalproject, content, flags=re.DOTALL)
        
        # 如果有变化，写回文件
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
    except (IOError, OSError) as e:
        print(f"Error processing {file_path}: {e}")
        return False

def check_files():
    """检查需要调整位置的文件"""
    print("Checking files that need position adjustment...")
    
    cmake_files = list(Path('.').rglob('*.cmake'))
    need_adjust_count = 0
    
    for file_path in cmake_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查是否有ExternalProject_Add、CMAKE_ARGS和CMAKE_POLICY_VERSION_MINIMUM
            if ('ExternalProject_Add' in content and 
                'CMAKE_ARGS' in content and 
                'CMAKE_POLICY_VERSION_MINIMUM' in content):
                
                # 检查版本参数是否不在第一行
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'CMAKE_ARGS' in line:
                        # 找到CMAKE_ARGS后的第一行
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip() and 'CMAKE_POLICY_VERSION_MINIMUM' not in lines[j]:
                                # 第一行不是版本参数，需要调整
                                print(f"Needs adjustment: {file_path}")
                                need_adjust_count += 1
                                break
                        break
        except (IOError, OSError) as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Found {need_adjust_count} files that need position adjustment")

def adjust_all_files():
    """调整所有文件的位置"""
    print("Adjusting CMAKE_POLICY_VERSION_MINIMUM position...")
    
    cmake_files = list(Path('.').rglob('*.cmake'))
    adjusted_count = 0
    
    for file_path in cmake_files:
        if adjust_cmake_policy_position(str(file_path)):
            print(f"Adjusted: {file_path}")
            adjusted_count += 1
    
    print(f"Adjusted {adjusted_count} files")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python adjust_cmake_policy_position.py <action>")
        print("Actions:")
        print("  check    - Check which files need position adjustment")
        print("  adjust   - Adjust CMAKE_POLICY_VERSION_MINIMUM position")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "check":
        check_files()
    elif action == "adjust":
        adjust_all_files()
    else:
        print("Invalid action")
        sys.exit(1)

if __name__ == "__main__":
    main() 