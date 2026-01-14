#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# -                       ACloudViewer Documentation Builder                 -
# ----------------------------------------------------------------------------
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""
Sphinx documentation builder for ACloudViewer

Usage examples:
    # Build documentation (Sphinx + Doxygen)
    python make_docs.py --sphinx --doxygen

    # Build for release (use version number instead of git hash)
    python make_docs.py --is_release --sphinx --doxygen

    # Build only Sphinx documentation
    python make_docs.py --sphinx

    # Build only C++ API documentation
    python make_docs.py --doxygen

    # Clean and rebuild
    python make_docs.py --clean --sphinx --doxygen
"""

import argparse
import importlib
import inspect
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _create_or_clear_dir(dir_path):
    """Create directory or clear it if it exists."""
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print(f"Removed directory {dir_path}")
    os.makedirs(dir_path)
    print(f"Created directory {dir_path}")


def _update_file(src, dst):
    """Copy file if destination doesn't exist or is older."""
    if Path(dst).exists():
        src_stat = os.stat(src)
        dst_stat = os.stat(dst)
        if src_stat.st_mtime - dst_stat.st_mtime <= 0:
            print(f"Copy skipped: {dst}")
            return
    print(f"Copy: {src}\n   -> {dst}")
    shutil.copy2(src, dst)


class DoxygenDocsBuilder:
    """
    Build C++ API documentation using Doxygen.
    
    Following approach:
    - Doxygen generates independent HTML documentation
    - HTML is copied to Sphinx output directory for unified navigation
    - XML is kept for optional Breathe integration (fallback)
    - No forced dependency between C++ docs and Sphinx
    
    Reference: https://github.com/isl-org/Open3D/blob/main/docs/make_docs.py
    """

    def __init__(self, html_output_dir, doxygen_output_dir=None):
        self.html_output_dir = html_output_dir
        self.doxygen_output_dir = doxygen_output_dir or "doxygen"

    def run(self):
        """Run Doxygen to generate C++ API documentation."""
        if not os.path.exists("Doxyfile"):
            print("⚠️  Doxyfile not found, skipping C++ API documentation")
            return

        print("=" * 70)
        print("🔨 Building C++ API Documentation (Doxygen)")
        print("=" * 70)
        print()
        
        # Use absolute path for doxygen output directory
        doxygen_abs_dir = os.path.abspath(self.doxygen_output_dir)
        _create_or_clear_dir(doxygen_abs_dir)
        print(f"📂 Doxygen output directory: {doxygen_abs_dir}")
        print()
        
        # Create a temporary Doxyfile with custom OUTPUT_DIRECTORY
        temp_doxyfile = os.path.join(doxygen_abs_dir, "Doxyfile.tmp")
        with open("Doxyfile", "r") as f_in:
            with open(temp_doxyfile, "w") as f_out:
                for line in f_in:
                    # Replace OUTPUT_DIRECTORY line
                    if line.strip().startswith("OUTPUT_DIRECTORY"):
                        f_out.write(f"OUTPUT_DIRECTORY       = {doxygen_abs_dir}\n")
                    else:
                        f_out.write(line)
        
        cmd = ["doxygen", temp_doxyfile]
        print(f'Command: "{" ".join(cmd)}"')
        print()
        
        try:
            subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
            print()
            
            # Copy mainpage images to Doxygen HTML output
            # These images are referenced in mainpage.dox via \htmlonly blocks
            # and won't be automatically copied by Doxygen
            mainpage_images = [
                'AbstractionLayers.png',
                'MainUI.png',
                'ICP-registration.png',
                'Reconstruction.png',
                'SemanticAnnotation.png'
            ]
            
            doxygen_html = os.path.join(doxygen_abs_dir, "html")
            if os.path.exists(doxygen_html):
                print("📸 Copying mainpage images to Doxygen output:")
                for img in mainpage_images:
                    src = Path('images') / img
                    dst = Path(doxygen_html) / img
                    if src.exists():
                        shutil.copy2(src, dst)
                        print(f"  ✓ {img} ({src.stat().st_size // 1024} KB)")
                    else:
                        print(f"  ⚠️  {img} not found in images/")
                print()
            
            # Copy Doxygen HTML output to final location as cpp_api/api/ subdirectory
            # This avoids overwriting Sphinx-generated pages (overview.html, quickstart.html, plugins.html)
            cpp_api_dir = os.path.join(self.html_output_dir, "html", "cpp_api")
            output_path = os.path.join(cpp_api_dir, "api")
            
            if os.path.exists(doxygen_html):
                # Remove old api/ subdirectory if it exists
                if os.path.exists(output_path):
                    print(f"🗑️  Removing old cpp_api/api/ directory: {output_path}")
                    shutil.rmtree(output_path)
                
                # Ensure cpp_api/ directory exists (for Sphinx-generated pages)
                os.makedirs(cpp_api_dir, exist_ok=True)
                
                print(f"📁 Copying Doxygen HTML: {doxygen_html} -> {output_path}")
                shutil.copytree(doxygen_html, output_path)
                
                index_file = Path(output_path) / "index.html"
                if index_file.exists():
                    print(f"✅ C++ API docs: {index_file.as_uri()}")
                    print()
                    print("Documentation available at:")
                    print(f"  - Relative: ../cpp_api/api/index.html")
                    print(f"  - Absolute: {output_path}/index.html")
                else:
                    print("⚠️  index.html not found in Doxygen output")
            else:
                print(f"⚠️  Doxygen HTML output not found: {doxygen_html}")
            
            # Note about XML output (optional, for Breathe fallback)
            doxygen_xml = os.path.join(doxygen_abs_dir, "xml")
            if os.path.exists(doxygen_xml):
                print(f"ℹ️  Doxygen XML also generated at {doxygen_xml}/ (for optional Breathe use)")
            
            print()
            print("=" * 70)
                
        except subprocess.CalledProcessError as e:
            print()
            print(f"❌ Doxygen build failed with exit code {e.returncode}")
            print("=" * 70)
            raise
        
        # Note: We keep the doxygen/ directory for:
        # 1. XML files (if Breathe integration is needed)
        # 2. Debugging (inspect raw Doxygen output)
        # It will be cleaned on next build by _create_or_clear_dir()


class PyAPIDocsBuilder:
    """
    Generate Python API *.rst files, per (sub) module, per class, per function.
    The file name is the full module name.

    E.g. If output_dir == "source/python_api", the following files are generated:
    source/python_api/cloudViewer.camera.rst
    source/python_api/cloudViewer.camera.PinholeCameraIntrinsic.rst
    ...
    
    """

    def __init__(self, output_dir="source/python_api", input_dir="source/python_api_in"):
        """
        input_dir: The input dir for custom rst files that override the
                   generated files.
        """
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.module_names = PyAPIDocsBuilder._get_documented_module_names()

    def generate_rst(self):
        print(f"🔨 Generating *.rst Python API docs in directory: {self.output_dir}")
        _create_or_clear_dir(self.output_dir)

        for module_name in self.module_names:
            try:
                module = self._try_import_module(module_name)
                self._generate_module_class_function_docs(module_name, module)
            except Exception as e:
                print(f"[Warning] Module {module_name} cannot be imported: {e}.")

    @staticmethod
    def _get_documented_module_names():
        """Reads the modules of the python api from documented_modules.txt"""
        module_names = []
        with open("documented_modules.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                m = re.match(r"^(cloudViewer\..*)\s*$", line)
                if m:
                    module_names.append(m.group(1))
        print("Documented modules:")
        for module_name in module_names:
            print("-", module_name)
        return module_names

    def _try_import_module(self, full_module_name):
        """Returns the module object for the given module path"""
        # Add multiple potential Python module paths (matches CMakeLists.txt)
        potential_paths = [
            os.path.join(os.path.dirname(__file__), "..", "build_app", "lib", "Release", "Python", "cuda"),
            os.path.join(os.path.dirname(__file__), "..", "build_app", "lib", "python_package"),
            os.path.join(os.path.dirname(__file__), "..", "build", "lib", "Release", "Python", "cuda"),
            os.path.join(os.path.dirname(__file__), "..", "build", "lib", "python_package"),
        ]
        for path in potential_paths:
            if os.path.exists(path):
                sys.path.insert(0, path)
        
        # Try to import cloudViewer (pybind variant first, then standard)
        try:
            import pybind as cloudViewer
            sys.modules['cloudViewer'] = cloudViewer
        except ImportError:
            import cloudViewer

        try:
            # Try to import directly
            module = importlib.import_module(full_module_name)
            return module
        except ImportError:
            # Traverse the module hierarchy
            current_module = cloudViewer
            for sub_module_name in full_module_name.split(".")[1:]:
                current_module = getattr(current_module, sub_module_name)
            return current_module

    def _generate_function_doc(self, full_module_name, function_name, output_path):
        out_string = ""
        out_string += "%s.%s" % (full_module_name, function_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name
        out_string += "\n\n" + ".. autofunction:: %s" % function_name
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    def _generate_class_doc(self, full_module_name, class_name, output_path):
        out_string = ""
        out_string += "%s.%s" % (full_module_name, class_name)
        out_string += "\n" + "-" * len(out_string)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name
        out_string += "\n\n" + ".. autoclass:: %s" % class_name
        out_string += "\n    :members:"
        out_string += "\n    :undoc-members:"
        out_string += "\n    :inherited-members:"
        out_string += "\n"

        with open(output_path, "w") as f:
            f.write(out_string)

    def _generate_module_doc(self, full_module_name, class_names, function_names, sub_module_names, sub_module_doc_path):
        class_names = sorted(class_names)
        function_names = sorted(function_names)
        out_string = ""
        out_string += full_module_name
        out_string += "\n" + "=" * len(full_module_name)
        out_string += "\n\n" + ".. currentmodule:: %s" % full_module_name

        if len(class_names) > 0:
            out_string += "\n\n**Classes**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for class_name in class_names:
                out_string += "\n    " + "%s" % (class_name,)
            out_string += "\n"

        if len(function_names) > 0:
            out_string += "\n\n**Functions**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for function_name in function_names:
                out_string += "\n    " + "%s" % (function_name,)
            out_string += "\n"

        if len(sub_module_names) > 0:
            out_string += "\n\n**Modules**"
            out_string += "\n\n.. autosummary::"
            out_string += "\n"
            for sub_module_name in sub_module_names:
                out_string += "\n    " + "%s" % (sub_module_name,)
            out_string += "\n"

        obj_names = class_names + function_names + sub_module_names
        if len(obj_names) > 0:
            out_string += "\n\n.. toctree::"
            out_string += "\n    :hidden:"
            out_string += "\n"
            for obj_name in obj_names:
                out_string += "\n    %s <%s.%s>" % (
                    obj_name,
                    full_module_name,
                    obj_name,
                )
            out_string += "\n"

        with open(sub_module_doc_path, "w") as f:
            f.write(out_string)

    def _generate_module_class_function_docs(self, full_module_name, module):
        print(f"  Generating docs for submodule: {full_module_name}")

        # Class docs
        class_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.isclass(obj[1]) and not obj[0].startswith('_')
        ]
        for class_name in class_names:
            file_name = "%s.%s.rst" % (full_module_name, class_name)
            output_path = os.path.join(self.output_dir, file_name)
            input_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(input_path):
                shutil.copyfile(input_path, output_path)
                continue
            self._generate_class_doc(full_module_name, class_name, output_path)

        # Function docs
        function_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.isroutine(obj[1]) and not obj[0].startswith('_')
        ]
        for function_name in function_names:
            file_name = "%s.%s.rst" % (full_module_name, function_name)
            output_path = os.path.join(self.output_dir, file_name)
            input_path = os.path.join(self.input_dir, file_name)
            if os.path.isfile(input_path):
                shutil.copyfile(input_path, output_path)
                continue
            self._generate_function_doc(full_module_name, function_name, output_path)

        # Submodule docs
        sub_module_names = [
            obj[0]
            for obj in inspect.getmembers(module)
            if inspect.ismodule(obj[1]) and not obj[0].startswith('_')
        ]
        documented_sub_module_names = [
            sub_module_name for sub_module_name in sub_module_names 
            if "%s.%s" % (full_module_name, sub_module_name) in self.module_names
        ]

        # Path
        sub_module_doc_path = os.path.join(self.output_dir, full_module_name + ".rst")
        input_path = os.path.join(self.input_dir, full_module_name + ".rst")
        if os.path.isfile(input_path):
            shutil.copyfile(input_path, sub_module_doc_path)
            return
        self._generate_module_doc(
            full_module_name,
            class_names,
            function_names,
            documented_sub_module_names,
            sub_module_doc_path,
        )


class JupyterDocsBuilder:
    """Copy Jupyter notebooks from jupyter/ to source/tutorial/."""

    def __init__(self, current_file_dir):
        self.current_file_dir = current_file_dir

    def run(self):
        """Copy Jupyter notebooks to tutorial directories."""
        print("📓 Copying Jupyter notebooks to tutorial directories...")
        
        nb_parent_src = Path(self.current_file_dir) / "jupyter"
        nb_parent_dst = Path(self.current_file_dir) / "source" / "tutorial"
        
        if not nb_parent_src.exists():
            print(f"⚠️  Jupyter directory not found: {nb_parent_src}")
            return
        
        # Get all subdirectories in jupyter/
        example_dirs = [
            name for name in os.listdir(nb_parent_src)
            if os.path.isdir(nb_parent_src / name)
        ]
        
        copied_count = 0
        for example_dir in example_dirs:
            in_dir = nb_parent_src / example_dir
            out_dir = nb_parent_dst / example_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all notebooks
            for nb_in_path in in_dir.glob("*.ipynb"):
                nb_out_path = out_dir / nb_in_path.name
                _update_file(nb_in_path, nb_out_path)
                copied_count += 1
            
            # Copy images directory if it exists
            if (in_dir / "images").exists():
                if (out_dir / "images").exists():
                    shutil.rmtree(out_dir / "images")
                print(f"Copy: {in_dir / 'images'}\n   -> {out_dir / 'images'}")
                shutil.copytree(in_dir / "images", out_dir / "images")
        
        # Copy helper Python files
        for py_file in ["cloudViewer_tutorial.py", "jupyter_run_all.py", "jupyter_strip_output.py"]:
            src_file = nb_parent_src / py_file
            if src_file.exists():
                dst_file = nb_parent_dst / py_file
                _update_file(src_file, dst_file)
        
        # Copy static images (like donation.png) to _static directory
        print("📸 Copying static images to _static directory...")
        static_src = Path(self.current_file_dir) / "images"
        static_dst = Path(self.current_file_dir) / "source" / "_static"
        
        if static_src.exists():
            static_dst.mkdir(parents=True, exist_ok=True)
            # Copy donation.png if it exists
            donation_img = static_src / "donation.png"
            if donation_img.exists():
                _update_file(donation_img, static_dst / "donation.png")
                print(f"  ✓ donation.png ({donation_img.stat().st_size // 1024} KB)")
        
        print(f"✅ Copied {copied_count} notebooks to tutorial directories")


class SphinxDocsBuilder:
    """Build main documentation and Python API using Sphinx."""

    def __init__(self, current_file_dir, html_output_dir, is_release, parallel):
        self.current_file_dir = current_file_dir
        self.html_output_dir = html_output_dir
        self.is_release = is_release
        self.parallel = parallel

    def run(self):
        """Run Sphinx to build HTML documentation."""
        print("🔨 Building documentation with Sphinx...")
        
        # Ensure pandoc is in PATH (for nbsphinx)
        try:
            import pypandoc
            pandoc_dir = os.path.dirname(pypandoc.get_pandoc_path())
            if pandoc_dir not in os.environ.get('PATH', ''):
                os.environ['PATH'] = f"{pandoc_dir}:{os.environ.get('PATH', '')}"
                print(f"📓 Added pandoc to PATH: {pandoc_dir}")
        except (ImportError, OSError):
            print("⚠️  pypandoc not found, nbsphinx may fail without pandoc")
        
        build_dir = os.path.join(self.html_output_dir, "html")
        nproc = multiprocessing.cpu_count() if self.parallel else 1
        print(f"Building docs with {nproc} processes")

        # Get version from project if available
        version = "3.9"
        release = "3.9.3"
        
        if self.is_release:
            print(f"Building docs for release: {release}")
            cmd = [
                "sphinx-build",
                "-j", str(nproc),
                "-b", "html",
                "-D", f"version={version}",
                "-D", f"release={release}",
                "source",  # Source directory
                build_dir,  # Output directory
            ]
        else:
            print("Building development documentation")
            cmd = [
                "sphinx-build",
                "-j", str(nproc),
                "-b", "html",
                "source",
                build_dir,
            ]

        print(f'Running: "{" ".join(cmd)}"')
        try:
            subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
            
            # Create symlink for convenience
            html_link = os.path.join(self.current_file_dir, "html")
            if not os.path.exists(html_link):
                os.symlink(build_dir, html_link)
                
            index_html = Path(build_dir) / "index.html"
            if index_html.exists():
                print(f"✅ Sphinx docs generated at {index_html.as_uri()}")
            else:
                print(f"✅ Sphinx docs generated at {build_dir}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Sphinx build failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Build ACloudViewer documentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--sphinx",
        action="store_true",
        default=False,
        help="Build Sphinx documentation (HTML)",
    )
    parser.add_argument(
        "--doxygen",
        action="store_true",
        default=False,
        help="Build Doxygen documentation (C++ API)",
    )
    parser.add_argument(
        "--is_release",
        action="store_true",
        default=False,
        help="Build for release (use version number instead of git hash)",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Enable parallel Sphinx build",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=False,
        help="Clean output directory before building",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for HTML documentation (default: docs/_out)",
    )
    parser.add_argument(
        "--doxygen-dir",
        type=str,
        default=None,
        help="Output directory for Doxygen temporary files (default: docs/doxygen or <output-dir>/../doxygen)",
    )

    args = parser.parse_args()

    # Get current directory
    pwd = os.path.dirname(os.path.realpath(__file__))
    os.chdir(pwd)

    print("=" * 70)
    print("  ACloudViewer Documentation Build System")
    print("=" * 70)
    print()

    # Output directory (use --output-dir if provided, otherwise default to docs/_out)
    if args.output_dir:
        html_output_dir = os.path.abspath(args.output_dir)
        print(f"📂 HTML output directory: {html_output_dir} (custom)")
    else:
        html_output_dir = os.path.join(pwd, "_out")
        print(f"📂 HTML output directory: {html_output_dir} (default)")
    
    # Doxygen directory (use --doxygen-dir if provided)
    if args.doxygen_dir:
        doxygen_output_dir = os.path.abspath(args.doxygen_dir)
        print(f"📂 Doxygen directory: {doxygen_output_dir} (custom)")
    elif args.output_dir:
        # If --output-dir is specified, put doxygen alongside it (build directory)
        doxygen_output_dir = os.path.join(os.path.dirname(html_output_dir), "doxygen")
        print(f"📂 Doxygen directory: {doxygen_output_dir} (alongside output)")
    else:
        # Default: put in source directory (backward compatible)
        doxygen_output_dir = os.path.join(pwd, "doxygen")
        print(f"📂 Doxygen directory: {doxygen_output_dir} (default)")
    print()
    
    # Clean or create output directory
    if args.clean or (args.sphinx or args.doxygen):
        _create_or_clear_dir(html_output_dir)

    # Build Doxygen documentation FIRST (Sphinx depends on it)
    if args.doxygen:
        ddb = DoxygenDocsBuilder(html_output_dir, doxygen_output_dir)
        ddb.run()
    else:
        print("ℹ️  Doxygen build disabled, use --doxygen to enable")

    # Generate Python API docs BEFORE Sphinx build
    if args.sphinx:
        pyapi = PyAPIDocsBuilder()
        pyapi.generate_rst()

    # Copy Jupyter notebooks BEFORE Sphinx build
    if args.sphinx:
        jdb = JupyterDocsBuilder(pwd)
        jdb.run()

    # Build Sphinx documentation AFTER Doxygen, PyAPI, and Jupyter copy
    if args.sphinx:
        sdb = SphinxDocsBuilder(pwd, html_output_dir, args.is_release, args.parallel)
        sdb.run()
    else:
        print("ℹ️  Sphinx build disabled, use --sphinx to enable")

    # If neither Sphinx nor Doxygen specified, show help
    if not args.sphinx and not args.doxygen:
        print()
        print("⚠️  No build targets specified!")
        print("Use --sphinx and/or --doxygen to build documentation")
        parser.print_help()
        return 1

    print()
    print("=" * 70)
    print("✅ Documentation build complete!")
    print("=" * 70)
    
    # Show output location
    if args.sphinx:
        html_dir = os.path.join(html_output_dir, "html")
        print(f"\n📂 Output: {html_dir}")
        print(f"\nTo view the documentation:")
        print(f"  cd {pwd}")
        print(f"  python3 -m http.server 8080 --directory {html_dir}")
        print(f"  Then open: http://localhost:8080")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

