#!/usr/bin/env python3
"""
修复ExternalProject_Add的CMake版本兼容性问题
"""

import sys
import re
from pathlib import Path

def fix_externalproject_cmake_version(file_path):
    """修复单个文件中的ExternalProject_Add版本问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 查找ExternalProject_Add调用
        # 匹配模式：ExternalProject_Add( 后面跟着参数
        pattern = r'ExternalProject_Add\s*\(\s*([^)]+?)\s*\)'
        
        def replace_externalproject(match):
            params = match.group(1)
            
            # 检查是否已经有CMAKE_POLICY_VERSION_MINIMUM参数
            if 'CMAKE_POLICY_VERSION_MINIMUM' in params:
                return match.group(0)  # 已经有参数，不需要修改
            
            # 检查是否有CMAKE_ARGS参数
            if 'CMAKE_ARGS' in params:
                # 在CMAKE_ARGS中添加版本参数
                cmake_args_pattern = r'CMAKE_ARGS\s+([^)]+?)(?=\s+[A-Z_]+:|$)'
                cmake_args_match = re.search(cmake_args_pattern, params, re.DOTALL)
                if cmake_args_match:
                    cmake_args = cmake_args_match.group(1)
                    # 将版本参数放在第一行
                    new_cmake_args = '\n        -DCMAKE_POLICY_VERSION_MINIMUM=3.5' + cmake_args
                    new_params = params.replace(cmake_args, new_cmake_args)
                    return f'ExternalProject_Add({new_params})'
                
                # CMAKE_ARGS存在但没有内容，添加参数
                new_params = params.replace('CMAKE_ARGS', 'CMAKE_ARGS\n        -DCMAKE_POLICY_VERSION_MINIMUM=3.5')
                return f'ExternalProject_Add({new_params})'
            
            # 没有CMAKE_ARGS，添加完整的CMAKE_ARGS参数
            # 找到最后一个参数的位置
            lines = params.split('\n')
            indent = ''
            for line in lines:
                if line.strip():
                    indent = line[:len(line) - len(line.lstrip())]
                    break
            
            # 在最后添加CMAKE_ARGS
            new_params = params.rstrip() + f'\n{indent}CMAKE_ARGS\n{indent}    -DCMAKE_POLICY_VERSION_MINIMUM=3.5'
            return f'ExternalProject_Add({new_params})'
        
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

def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("Usage: python fix_externalproject_cmake_version.py <action>")
        print("Actions:")
        print("  check    - Check which files need CMake version fix")
        print("  fix      - Fix CMake version compatibility issues")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "check":
        print("Checking files that need CMake version fix...")
        
        # 查找所有.cmake文件
        cmake_files = list(Path('.').rglob('*.cmake'))
        need_fix_count = 0
        
        for file_path in cmake_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有ExternalProject_Add但没有CMAKE_POLICY_VERSION_MINIMUM
                if 'ExternalProject_Add' in content and 'CMAKE_POLICY_VERSION_MINIMUM' not in content:
                    print(f"Needs fix: {file_path}")
                    need_fix_count += 1
            except (IOError, OSError) as e:
                print(f"Error reading {file_path}: {e}")
        
        print(f"Found {need_fix_count} files that need CMake version fix")
        
    elif action == "fix":
        print("Fixing CMake version compatibility issues...")
        
        # 查找所有.cmake文件
        cmake_files = list(Path('.').rglob('*.cmake'))
        fixed_count = 0
        
        for file_path in cmake_files:
            if fix_externalproject_cmake_version(str(file_path)):
                print(f"Fixed: {file_path}")
                fixed_count += 1
        
        print(f"Fixed {fixed_count} files")
        
    else:
        print("Invalid action")
        sys.exit(1)

if __name__ == "__main__":
    main() 