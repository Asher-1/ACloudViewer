#!/usr/bin/env python3
"""
清理CMake文件中的多余空行
"""

import sys
from pathlib import Path

def cleanup_empty_lines(file_path):
    """清理单个文件中的多余空行"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 移除多余的空行
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            is_empty = line.strip() == ''
            
            # 如果是空行且前一行也是空行，则跳过
            if is_empty and prev_empty:
                continue
            
            cleaned_lines.append(line)
            prev_empty = is_empty
        
        # 移除文件末尾的空行
        while cleaned_lines and cleaned_lines[-1].strip() == '':
            cleaned_lines.pop()
        
        # 如果有变化，写回文件
        original_content = ''.join(lines)
        new_content = ''.join(cleaned_lines)
        
        if new_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        
        return False
    except (IOError, OSError) as e:
        print(f"Error processing {file_path}: {e}")
        return False

def check_excessive_empty_lines():
    """检查有多余空行的文件"""
    print("Checking files with excessive empty lines...")
    
    cmake_files = list(Path('.').rglob('CMakeLists.txt'))
    excessive_count = 0
    
    for file_path in cmake_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # 检查是否有连续的空行
            consecutive_empty = 0
            has_excessive = False
            
            for line in lines:
                if line.strip() == '':
                    consecutive_empty += 1
                    if consecutive_empty > 1:
                        has_excessive = True
                        break
                else:
                    consecutive_empty = 0
            
            if has_excessive:
                print(f"Excessive empty lines: {file_path}")
                excessive_count += 1
        except (IOError, OSError) as e:
            print(f"Error reading {file_path}: {e}")
    
    print(f"Found {excessive_count} files with excessive empty lines")

def cleanup_all_files():
    """清理所有文件的多余空行"""
    print("Cleaning up excessive empty lines...")
    
    cmake_files = list(Path('.').rglob('CMakeLists.txt'))
    cleaned_count = 0
    
    for file_path in cmake_files:
        if cleanup_empty_lines(str(file_path)):
            print(f"Cleaned: {file_path}")
            cleaned_count += 1
    
    print(f"Cleaned {cleaned_count} files")

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python cleanup_empty_lines.py <action>")
        print("Actions:")
        print("  check    - Check which files have excessive empty lines")
        print("  cleanup  - Clean up excessive empty lines")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "check":
        check_excessive_empty_lines()
    elif action == "cleanup":
        cleanup_all_files()
    else:
        print("Invalid action")
        sys.exit(1)

if __name__ == "__main__":
    main() 