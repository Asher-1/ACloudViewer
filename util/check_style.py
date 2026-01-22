# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import subprocess
import re
import shutil
import argparse
from pathlib import Path
import multiprocessing
from functools import partial
import time
import sys

PYTHON_FORMAT_DIRS = [
    "examples",
    "python",
    "util",
]

JUPYTER_FORMAT_DIRS = [
    "examples",
]

# Note: also modify CPP_FORMAT_DIRS in check_cpp_style.cmake.
CPP_FORMAT_DIRS = [
    "core",
    "libs",
    "app",
    "plugins",
    "examples",
]

import yapf
import nbformat


class CppFormatter:

    standard_header = """// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
"""

    def __init__(self, file_paths, clang_format_bin):
        self.file_paths = file_paths
        self.clang_format_bin = clang_format_bin

    @staticmethod
    def _check_style(file_path, clang_format_bin):
        """
        Returns (true, true) if (style, header) is valid.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                is_valid_header = (f.read(len(CppFormatter.standard_header)) ==
                                   CppFormatter.standard_header)
        except Exception as exp:
            print(f"Error reading file header {file_path}: {exp}")
            is_valid_header = False

        cmd = [
            clang_format_bin,
            "-style=file",
            "-output-replacements-xml",
            file_path,
        ]
        result = subprocess.check_output(cmd).decode("utf-8")
        if "<replacement " in result:
            is_valid_style = False
        else:
            is_valid_style = True
        return (is_valid_style, is_valid_header)

    @staticmethod
    def _apply_style(file_path, clang_format_bin):
        cmd = [
            clang_format_bin,
            "-style=file",
            "-i",
            file_path,
        ]
        subprocess.check_output(cmd)

    @staticmethod
    def _apply_header(file_path):
        """
        Safely replace the file header with the standard header.
        This method carefully identifies and replaces only the license header block.
        """
        try:
            # Create backup
            import shutil
            backup_path = file_path + '.backup'
            shutil.copy2(file_path, backup_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]

            if not content.strip():
                # Empty file, just add header
                new_content = CppFormatter.standard_header
            else:
                lines = content.split('\n')

                # First, check if there are any comment lines at the beginning
                # that look like license headers
                first_code_line = 0
                has_header_comment = False
                in_block_comment = False

                for i, line in enumerate(lines):
                    stripped = line.strip()

                    # Track block comment state
                    if '/*' in stripped:
                        in_block_comment = True

                    # Empty lines
                    if not stripped:
                        if not in_block_comment:
                            continue
                        else:
                            # Empty line inside block comment, continue
                            continue

                    # Check if we're in a block comment
                    if in_block_comment:
                        # Check if this looks like a license/header comment
                        if any(keyword in line for keyword in [
                                'Copyright', 'copyright', 'LICENSE', 'License',
                                'CloudViewer', 'CLOUDVIEWER', 'EDF R&D',
                                'GNU General Public License', 'GPL', 'BSD',
                                'MIT', 'Apache', 'SPDX', 'All rights reserved',
                                'This program is free software', 'ParaView',
                                'Kitware', 'Sandia', 'CLOUDCOMPARE'
                        ]):
                            has_header_comment = True

                        # Check if block comment ends on this line
                        if '*/' in stripped:
                            in_block_comment = False
                        continue

                    # Single line comment (// or decorative lines like // # or // -)
                    if stripped.startswith('//') or stripped.startswith('-'):
                        # Check if this looks like a license/header comment
                        if any(keyword in stripped for keyword in [
                                'Copyright', 'copyright', 'LICENSE', 'License',
                                'CloudViewer', 'CLOUDVIEWER', 'EDF R&D',
                                'GNU General Public License', 'GPL', 'BSD',
                                'MIT', 'Apache', 'SPDX', 'All rights reserved',
                                'This program is free software', 'ParaView',
                                'Kitware', 'Sandia', '########', 'CLOUDCOMPARE'
                        ]):
                            has_header_comment = True
                        # Continue to next line (still in comment section)
                        continue

                    # Check for decorative comment lines like "// #" or "// ##########"
                    if stripped.startswith('// #'):
                        # This is a decorative comment line
                        has_header_comment = True
                        continue
                    else:
                        # Non-comment, non-empty line = actual code
                        first_code_line = i
                        break

                # If we found header comments, replace them
                # If not, just prepend the header
                if has_header_comment and first_code_line > 0:
                    # Replace existing header comments
                    after_header = lines[first_code_line:]
                    header_lines = CppFormatter.standard_header.rstrip(
                        '\n').split('\n')
                    if after_header and after_header[0].strip():
                        header_lines.append('')  # Add blank line after header
                    new_lines = header_lines + after_header
                    new_content = '\n'.join(new_lines)
                else:
                    # No header comments found, just prepend standard header
                    header_lines = CppFormatter.standard_header.rstrip(
                        '\n').split('\n')
                    new_lines = header_lines + [''] + lines
                    new_content = '\n'.join(new_lines)

            # Validate the new content by trying to parse it (basic check)
            if file_path.endswith(('.cpp', '.cc', '.cxx')):
                # For C++ files, just check it's not empty and has some basic structure
                if not new_content.strip():
                    raise ValueError("Generated content is empty")

            # Write the new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            # Remove backup if successful
            import os
            os.remove(backup_path)

        except Exception as exp:
            print(f"Error applying header to {file_path}: {exp}")
            # Restore from backup if it exists
            try:
                import os
                backup_path = file_path + '.backup'
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, file_path)
                    os.remove(backup_path)
            except:
                pass

    def run(self, apply, no_parallel, verbose):
        num_procs = multiprocessing.cpu_count() if not no_parallel else 1
        action_name = "Applying C++/CUDA style" if apply else "Checking C++/CUDA style"
        print(f"{action_name} ({num_procs} process{'es'[:2*num_procs^2]})")
        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print(f"> {file_path}")

        start_time = time.time()
        with multiprocessing.Pool(num_procs) as pool:
            is_valid_files = pool.map(
                partial(self._check_style,
                        clang_format_bin=self.clang_format_bin),
                self.file_paths)

        changed_files = []
        wrong_header_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            is_valid_style = is_valid[0]
            is_valid_header = is_valid[1]
            if not is_valid_style:
                changed_files.append(file_path)
                if apply:
                    self._apply_style(file_path, self.clang_format_bin)
            if not is_valid_header:
                if apply:
                    # Auto-replace header when applying
                    self._apply_header(file_path)
                    changed_files.append(file_path)
                else:
                    wrong_header_files.append(file_path)
        print(f"{action_name} took {time.time() - start_time:.2f}s")

        return (changed_files, wrong_header_files)


class PythonFormatter:

    standard_header = """# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""

    def __init__(self, file_paths, style_config):
        self.file_paths = file_paths
        self.style_config = style_config

    @staticmethod
    def _check_style(file_path, style_config):
        """
        Returns (true, true) if (style, header) is valid.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                is_valid_header = (len(content) == 0 or content.startswith(
                    PythonFormatter.standard_header))
        except Exception as exp:
            print(f"Error reading file header {file_path}: {exp}")
            is_valid_header = False

        try:
            _, _, changed = yapf.yapflib.yapf_api.FormatFile(
                file_path, style_config=style_config, in_place=False)
            return (not changed, is_valid_header)
        except Exception as exp:
            print(f"Error checking Python style for {file_path}: {exp}")
            # If we can't check the style due to syntax errors, consider it invalid
            return (False, is_valid_header)

    @staticmethod
    def _apply_style(file_path, style_config):
        _, _, _ = yapf.yapflib.yapf_api.FormatFile(file_path,
                                                   style_config=style_config,
                                                   in_place=True)

    @staticmethod
    def _apply_header(file_path):
        """
        Safely replace the file header with the standard header.
        This method carefully identifies and replaces only the license header block.
        """
        try:
            # Create backup
            import shutil
            import os
            backup_path = file_path + '.backup'
            shutil.copy2(file_path, backup_path)

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            if not content.strip():
                # Empty file, just add header
                new_content = PythonFormatter.standard_header
            else:
                lines = content.split('\n')
                header_start = -1
                header_end = -1

                # Look for existing CloudViewer header pattern
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if ('CloudViewer:' in stripped and 'www.cloudViewer.org' in stripped) or \
                       ('Copyright (c)' in stripped and 'www.cloudViewer.org' in stripped):
                        # Found CloudViewer header, find the start of this comment block
                        header_start = i
                        # Look backwards to find the start of the comment block
                        for j in range(i - 1, -1, -1):
                            prev_line = lines[j].strip()
                            if prev_line.startswith('#') or prev_line == '' or \
                               prev_line.startswith('-'):
                                header_start = j
                            else:
                                break
                        break

                if header_start >= 0:
                    # Found existing header, find its end
                    for i in range(header_start, len(lines)):
                        stripped = lines[i].strip()
                        if stripped and not stripped.startswith('#') and \
                           not stripped.startswith('-') and stripped != '':
                            header_end = i
                            break
                    else:
                        header_end = len(lines)

                    # Replace the existing header
                    before_header = lines[:header_start] if header_start > 0 else []
                    after_header = lines[header_end:]

                    # Clean up empty lines before header
                    while before_header and not before_header[-1].strip():
                        before_header.pop()

                    # Ensure proper spacing
                    header_lines = PythonFormatter.standard_header.rstrip(
                        '\n').split('\n')
                    if after_header and after_header[0].strip():
                        header_lines.append('')  # Add blank line after header

                    new_lines = before_header + header_lines + after_header
                    new_content = '\n'.join(new_lines)
                else:
                    # No existing CloudViewer header found, add at the beginning
                    # But be careful not to break shebang lines
                    if not lines:
                        # Empty file, just add header
                        new_content = PythonFormatter.standard_header
                    elif lines[0].strip().startswith('#!'):
                        # Keep shebang line, then check for old headers after it
                        content_start = 1  # Start after shebang
                        in_potential_header = True

                        for i in range(1, len(lines)):
                            stripped = lines[i].strip()
                            if not stripped:  # Empty line
                                continue
                            elif stripped.startswith(
                                    '#') and in_potential_header:
                                # This might be part of an old header, continue looking
                                continue
                            elif stripped.startswith(
                                    '"""') or stripped.startswith("'''"):
                                # Docstring - this is actual content
                                content_start = i
                                in_potential_header = False
                                break
                            elif not stripped.startswith('#'):
                                # Non-comment line - this is actual content
                                content_start = i
                                in_potential_header = False
                                break

                        header_lines = PythonFormatter.standard_header.rstrip(
                            '\n').split('\n')
                        new_lines = [lines[0]] + header_lines + [
                            ''
                        ] + lines[content_start:]
                        new_content = '\n'.join(new_lines)
                    else:
                        # Check if there are any existing comment blocks at the top that might be old headers
                        content_start = 0
                        in_potential_header = True

                        for i, line in enumerate(lines):
                            stripped = line.strip()
                            if not stripped:  # Empty line
                                continue
                            elif stripped.startswith(
                                    '#') and in_potential_header:
                                # This might be part of an old header, continue looking
                                continue
                            elif stripped.startswith(
                                    '"""') or stripped.startswith("'''"):
                                # Docstring - this is actual content
                                content_start = i
                                in_potential_header = False
                                break
                            elif not stripped.startswith('#'):
                                # Non-comment line - this is actual content
                                content_start = i
                                in_potential_header = False
                                break

                        # Add standard header before the actual content
                        header_lines = PythonFormatter.standard_header.rstrip(
                            '\n').split('\n')
                        if content_start > 0:
                            # There were some comments/empty lines at the top, replace them
                            new_lines = header_lines + [''
                                                       ] + lines[content_start:]
                        else:
                            # No comments at top, just add header
                            new_lines = header_lines + [''] + lines
                        new_content = '\n'.join(new_lines)

            # Validate the new content by trying to parse it
            try:
                import ast
                ast.parse(new_content)
            except SyntaxError as e:
                raise ValueError(f"Generated content has syntax error: {e}")

            # Write the new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            # Remove backup if successful
            os.remove(backup_path)

        except Exception as exp:
            print(f"Error applying header to {file_path}: {exp}")
            # Restore from backup if it exists
            try:
                import os
                import shutil
                backup_path = file_path + '.backup'
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, file_path)
                    os.remove(backup_path)
            except:
                pass

    def run(self, apply, no_parallel, verbose):
        num_procs = multiprocessing.cpu_count() if not no_parallel else 1
        action_name = "Applying Python style" if apply else "Checking Python style"
        print(f"{action_name} ({num_procs} process{'es'[:2*num_procs^2]})")

        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print(f"> {file_path}")

        start_time = time.time()
        with multiprocessing.Pool(num_procs) as pool:
            is_valid_files = pool.map(
                partial(self._check_style, style_config=self.style_config),
                self.file_paths)

        changed_files = []
        wrong_header_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            is_valid_style = is_valid[0]
            is_valid_header = is_valid[1]
            if not is_valid_style:
                changed_files.append(file_path)
                if apply:
                    self._apply_style(file_path, self.style_config)
            if not is_valid_header:
                if apply:
                    # Auto-replace header when applying
                    self._apply_header(file_path)
                    changed_files.append(file_path)
                else:
                    wrong_header_files.append(file_path)

        print(f"{action_name} took {time.time() - start_time:.2f}s")
        return (changed_files, wrong_header_files)


class JupyterFormatter:

    def __init__(self, file_paths, style_config):
        self.file_paths = file_paths
        self.style_config = style_config

    @staticmethod
    def _is_valid_python_code(src):
        """
        Check if the source code is valid Python that can be formatted by yapf.
        Returns True if it's valid Python code, False otherwise.
        """
        if not src.strip():
            return False

        lines = [line.strip() for line in src.split("\n") if line.strip()]

        # Skip if contains shell commands or magic commands
        for line in lines:
            if line.startswith(('!', '%', '%%', '?', '??')):
                return False

        # Try to parse as Python to verify it's valid syntax
        try:
            import ast
            ast.parse(src)
            return True
        except SyntaxError:
            return False

    @staticmethod
    def _check_or_apply_style(file_path, style_config, apply):
        """
        Returns true if style is valid.

        Since there are common code for check and apply style, the two functions
        are merged into one.
        """
        # Ref: https://gist.github.com/oskopek/496c0d96c79fb6a13692657b39d7c709
        with open(file_path, "r", encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=nbformat.NO_CONVERT)
        nbformat.validate(notebook)

        changed = False
        for cell in notebook.cells:
            if cell["cell_type"] != "code":
                continue
            src = cell["source"]

            # Skip if empty
            if not src.strip():
                continue

            # Check if first line has "# noqa" comment
            lines = src.split("\n")
            if lines and "# noqa" in lines[0]:
                continue

            # Only format cells that contain valid Python code
            if not JupyterFormatter._is_valid_python_code(src):
                continue

            # yapf will puts a `\n` at the end of each cell, and if this is the
            # only change, cell_changed is still False.
            try:
                formatted_src, cell_changed = yapf.yapflib.yapf_api.FormatCode(
                    src, style_config=style_config)
                if formatted_src.endswith("\n"):
                    formatted_src = formatted_src[:-1]

                # Check if content actually changed (not just yapf's internal flag)
                if src != formatted_src:
                    if apply:
                        cell["source"] = formatted_src
                    changed = True
            except Exception as e:
                # This should not happen since we validated the code above
                # But keep as safety net
                print(f"Warning: Failed to format cell in {file_path}: {e}")
                continue

        if apply and changed:
            with open(file_path, "w", encoding='utf-8') as f:
                nbformat.write(notebook, f, version=nbformat.NO_CONVERT)

        return not changed

    def run(self, apply, no_parallel, verbose):
        num_procs = multiprocessing.cpu_count() if not no_parallel else 1
        action_name = "Applying Jupyter style" if apply else "Checking Jupyter style"
        print(f"{action_name} ({num_procs} process{'es'[:2*num_procs^2]})")

        if verbose:
            print("To format:")
            for file_path in self.file_paths:
                print(f"> {file_path}")

        start_time = time.time()
        with multiprocessing.Pool(num_procs) as pool:
            is_valid_files = pool.map(
                partial(self._check_or_apply_style,
                        style_config=self.style_config,
                        apply=False), self.file_paths)

        changed_files = []
        for is_valid, file_path in zip(is_valid_files, self.file_paths):
            if not is_valid:
                changed_files.append(file_path)
                if apply:
                    self._check_or_apply_style(file_path,
                                               style_config=self.style_config,
                                               apply=True)
        print(f"{action_name} took {time.time() - start_time:.2f}s")

        return changed_files


def _glob_files(directories, extensions):
    """
    Find files with certain extensions in directories recursively.

    Args:
        directories: list of directories, relative to the root Open3D repo directory.
        extensions: list of extensions, e.g. ["cpp", "h"].

    Return:
        List of file paths.
    """
    pwd = Path(__file__).resolve().parent
    open3d_root_dir = pwd.parent

    file_paths = []
    for directory in directories:
        directory = open3d_root_dir / directory
        for extension in extensions:
            extension_regex = "*." + extension
            file_paths.extend(directory.rglob(extension_regex))
    file_paths = [
        str(file_path) for file_path in file_paths if file_path.name[0] != '.'
    ]
    file_paths = sorted(list(set(file_paths)))
    return file_paths


def _find_clang_format():
    """
    Returns (bin_path, version) to clang-format, throws exception
    otherwise.
    """

    def parse_version(bin_path):
        """
        Get clang-format version string. Returns None if parsing fails.
        """
        version_str = subprocess.check_output([bin_path, "--version"
                                              ]).decode("utf-8").strip()
        match = re.match("^.*clang-format version ([0-9.]*).*$", version_str)
        return match.group(1) if match else None

    bin_path = shutil.which("clang-format")
    if bin_path is not None:
        bin_version = parse_version(bin_path)
        return bin_path, bin_version

    raise RuntimeError(
        "clang-format version not found. Please install with "
        "'pip install -c python/requirements_style.txt clang-format'")


def _filter_files(files, ignored_patterns):
    return [
        file for file in files if not any(
            [ignored_pattern in file for ignored_pattern in ignored_patterns])
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--apply",
        dest="apply",
        action="store_true",
        default=False,
        help="Apply style to files in-place.",
    )
    parser.add_argument(
        "--no_parallel",
        dest="no_parallel",
        action="store_true",
        default=False,
        help="Disable parallel execution.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="If true, prints file names while formatting.",
    )
    args = parser.parse_args()

    # Check formatting libs
    clang_format_bin, clang_format_version = _find_clang_format()
    print(f"Using clang-format {clang_format_version} ({clang_format_bin})")
    print(f"Using yapf {yapf.__version__} ({yapf.__file__})")
    print(f"Using nbformat {nbformat.__version__} ({nbformat.__file__})")
    pwd = Path(__file__).resolve().parent
    python_style_config = str(pwd.parent / ".style.yapf")

    cpp_ignored_files = [
        'cpp/open3d/visualization/shader/Shader.h',
        # Third-party libraries in libs/Reconstruction/lib/
        'libs/Reconstruction/lib/PoissonRecon/',
        'libs/Reconstruction/lib/SiftGPU/',
        'libs/Reconstruction/lib/FLANN/',
        'libs/Reconstruction/lib/Graclus/',
        'libs/Reconstruction/lib/LSD/',
        'libs/Reconstruction/lib/PBA/',
        'libs/Reconstruction/lib/VLFeat/',
        'libs/Reconstruction/lib/SQLite/',
        # Third-party libraries in plugins/extern/
        'plugins/core/IO/qPhotoscanIO/extern/',
        'plugins/core/IO/qE57IO/extern/',
        'plugins/core/Standard/qPoissonRecon/extern/',
        'plugins/core/Standard/qHoughNormals/extern/',
        'plugins/core/Standard/qAnimation/extern/',
        'plugins/core/Standard/qCork/extern/',
        # Other external libraries
        'libs/CV_io/extern/',
    ]
    cpp_files = _glob_files(CPP_FORMAT_DIRS,
                            ["h", "cpp", "cuh", "cu", "isph", "ispc", "h.in"])
    cpp_files = _filter_files(cpp_files, cpp_ignored_files)

    # Check or apply style
    cpp_formatter = CppFormatter(cpp_files, clang_format_bin=clang_format_bin)
    python_formatter = PythonFormatter(_glob_files(PYTHON_FORMAT_DIRS, ["py"]),
                                       style_config=python_style_config)
    jupyter_formatter = JupyterFormatter(_glob_files(JUPYTER_FORMAT_DIRS,
                                                     ["ipynb"]),
                                         style_config=python_style_config)

    changed_files = []
    wrong_header_files = []
    changed_files_cpp, wrong_header_files_cpp = cpp_formatter.run(
        apply=args.apply, no_parallel=args.no_parallel, verbose=args.verbose)
    changed_files.extend(changed_files_cpp)
    wrong_header_files.extend(wrong_header_files_cpp)

    changed_files_python, wrong_header_files_python = python_formatter.run(
        apply=args.apply, no_parallel=args.no_parallel, verbose=args.verbose)
    changed_files.extend(changed_files_python)
    wrong_header_files.extend(wrong_header_files_python)

    changed_files.extend(
        jupyter_formatter.run(apply=args.apply,
                              no_parallel=args.no_parallel,
                              verbose=args.verbose))

    if len(changed_files) == 0 and len(wrong_header_files) == 0:
        print("All files passed style check")
        exit(0)

    if args.apply:
        if len(changed_files) != 0:
            print("Style applied to the following files:")
            print("\n".join(changed_files))
        if len(wrong_header_files) != 0:
            print("Please correct license header *manually* in the following "
                  "files (see util/check_style.py for the standard header):")
            print("\n".join(wrong_header_files))
            exit(1)
    else:
        error_files_no_duplicates = list(set(changed_files +
                                             wrong_header_files))
        if len(error_files_no_duplicates) != 0:
            print("Style error found in the following files:")
            print("\n".join(error_files_no_duplicates))
            exit(1)


if __name__ == "__main__":
    main()
