#!/usr/bin/env python3
"""
Docstring Auto-Fix Tool
========================
This script automatically fixes common docstring format issues in C++ pybind11
bindings to make them compatible with Sphinx RST parser.

Usage:
    python fix_docstrings.py <file_or_directory> [--dry-run] [--backup]

Examples:
    # Fix a single file
    python fix_docstrings.py python/pybind/geometry/pointcloud.cpp

    # Fix all files in a module
    python fix_docstrings.py python/pybind/geometry/ --backup

    # Preview changes without modifying files
    python fix_docstrings.py python/pybind/ --dry-run
"""

import re
import sys
import argparse
from pathlib import Path
from typing import List, Tuple
import shutil

# ANSI color codes
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color


class DocstringFixer:
    def __init__(self, dry_run=False, backup=False):
        self.dry_run = dry_run
        self.backup = backup
        self.fixes_count = 0
        self.files_modified = 0
        
    def fix_file(self, filepath: Path) -> bool:
        """Fix docstrings in a single C++ file."""
        print(f"\n{Colors.BLUE}Processing:{Colors.NC} {filepath}")
        
        try:
            content = filepath.read_text(encoding='utf-8')
            original_content = content
            
            # Apply fixes
            content = self.fix_unescaped_asterisks(content)
            content = self.fix_args_indentation(content)
            content = self.fix_returns_indentation(content)
            content = self.fix_example_blocks(content)
            content = self.fix_inline_code_formatting(content)
            content = self.fix_broken_emphasis(content)
            
            if content != original_content:
                if not self.dry_run:
                    if self.backup:
                        backup_path = filepath.with_suffix('.cpp.bak')
                        shutil.copy2(filepath, backup_path)
                        print(f"  {Colors.GREEN}✓{Colors.NC} Backup created: {backup_path.name}")
                    
                    filepath.write_text(content, encoding='utf-8')
                    print(f"  {Colors.GREEN}✓{Colors.NC} File modified")
                else:
                    print(f"  {Colors.YELLOW}[DRY RUN]{Colors.NC} Would modify file")
                
                self.files_modified += 1
                return True
            else:
                print(f"  {Colors.BLUE}○{Colors.NC} No changes needed")
                return False
                
        except Exception as e:
            print(f"  {Colors.RED}✗{Colors.NC} Error: {e}")
            return False
    
    def fix_unescaped_asterisks(self, content: str) -> str:
        """Fix unescaped asterisks in docstrings that cause RST emphasis issues."""
        
        # Pattern to match .def() with docstrings
        def_pattern = r'(\.def\([^"]*")((?:[^"\\]|\\.)*)(")'
        
        def replace_asterisks(match):
            prefix = match.group(1)
            docstring = match.group(2)
            suffix = match.group(3)
            
            # Replace unescaped single asterisks with escaped ones
            # But preserve ** (strong emphasis) and already escaped \*
            fixed = re.sub(r'(?<!\\)(?<!\*)\*(?!\*)(?!\s)', r'\\*', docstring)
            
            if fixed != docstring:
                self.fixes_count += 1
                
            return prefix + fixed + suffix
        
        return re.sub(def_pattern, replace_asterisks, content)
    
    def fix_args_indentation(self, content: str) -> str:
        """Fix improper indentation in Args sections."""
        
        # Find docstrings with Args: but missing proper indentation
        pattern = r'(\.def\([^"]*"(?:[^"\\]|\\.)*Args:\\n)([A-Za-z])'
        
        def add_indentation(match):
            self.fixes_count += 1
            return match.group(1) + '    ' + match.group(2)
        
        return re.sub(pattern, add_indentation, content)
    
    def fix_returns_indentation(self, content: str) -> str:
        """Fix improper indentation in Returns sections."""
        
        pattern = r'(\.def\([^"]*"(?:[^"\\]|\\.)*Returns:\\n)([A-Za-z])'
        
        def add_indentation(match):
            self.fixes_count += 1
            return match.group(1) + '    ' + match.group(2)
        
        return re.sub(pattern, add_indentation, content)
    
    def fix_example_blocks(self, content: str) -> str:
        """Fix Example: blocks to use proper RST code block syntax (::)."""
        
        # Example: followed by code needs ::
        pattern = r'(\.def\([^"]*"(?:[^"\\]|\\.)*Example:)([^:])'
        
        def add_double_colon(match):
            self.fixes_count += 1
            return match.group(1) + ':' + match.group(2)
        
        return re.sub(pattern, add_double_colon, content)
    
    def fix_inline_code_formatting(self, content: str) -> str:
        """Fix inline code to use double backticks instead of single quotes."""
        
        # Find patterns like 'True' or 'False' and replace with ``True``
        pattern = r"(\.def\([^\"]*\"(?:[^\"\\]|\\.)*)\'(True|False|None)\'"
        
        def replace_with_backticks(match):
            self.fixes_count += 1
            return match.group(1) + '``' + match.group(2) + '``'
        
        return re.sub(pattern, replace_with_backticks, content)
    
    def fix_broken_emphasis(self, content: str) -> str:
        """Fix broken emphasis markers (single asterisk at end of string)."""
        
        pattern = r'(\.def\([^"]*"(?:[^"\\]|\\.)*)(?<!\\)\*(["\s,])'
        
        def escape_trailing_asterisk(match):
            self.fixes_count += 1
            return match.group(1) + r'\\*' + match.group(2)
        
        return re.sub(pattern, escape_trailing_asterisk, content)
    
    def process_path(self, path: Path):
        """Process a file or directory."""
        if path.is_file():
            if path.suffix == '.cpp':
                self.fix_file(path)
        elif path.is_dir():
            cpp_files = list(path.rglob('*.cpp'))
            total = len(cpp_files)
            
            print(f"\n{Colors.BLUE}Found {total} C++ files in {path}{Colors.NC}")
            
            for i, cpp_file in enumerate(cpp_files, 1):
                print(f"\n[{i}/{total}]", end=' ')
                self.fix_file(cpp_file)
        else:
            print(f"{Colors.RED}Error:{Colors.NC} {path} is not a valid file or directory")
    
    def print_summary(self):
        """Print summary of fixes applied."""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
        print(f"{Colors.BLUE}Summary{Colors.NC}")
        print(f"{Colors.BLUE}{'='*70}{Colors.NC}")
        print(f"Files modified: {self.files_modified}")
        print(f"Total fixes applied: {self.fixes_count}")
        
        if self.dry_run:
            print(f"\n{Colors.YELLOW}This was a dry run. No files were actually modified.{Colors.NC}")
            print(f"Run without --dry-run to apply changes.")
        
        print()


def main():
    parser = argparse.ArgumentParser(
        description='Automatically fix common docstring format issues in C++ pybind11 bindings.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fix a single file
  python fix_docstrings.py python/pybind/geometry/pointcloud.cpp

  # Fix all files in a directory with backup
  python fix_docstrings.py python/pybind/geometry/ --backup

  # Preview changes without modifying files
  python fix_docstrings.py python/pybind/ --dry-run
        """
    )
    
    parser.add_argument('path', type=str,
                        help='Path to C++ file or directory to process')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without modifying files')
    parser.add_argument('--backup', action='store_true',
                        help='Create .bak backup files before modifying')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"{Colors.RED}Error:{Colors.NC} Path does not exist: {path}")
        sys.exit(1)
    
    print(f"{Colors.BLUE}╔═══════════════════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.BLUE}║         ACloudViewer Docstring Auto-Fix Tool                 ║{Colors.NC}")
    print(f"{Colors.BLUE}╚═══════════════════════════════════════════════════════════════╝{Colors.NC}")
    
    fixer = DocstringFixer(dry_run=args.dry_run, backup=args.backup)
    fixer.process_path(path)
    fixer.print_summary()


if __name__ == '__main__':
    main()
