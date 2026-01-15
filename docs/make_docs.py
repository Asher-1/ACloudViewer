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
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

import certifi
import nbconvert
import nbformat


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


class PyExampleDocsBuilder:
    """
    Generate Python examples *.rst files.
    """

    def __init__(self, input_dir, pwd, output_dir="source/python_example"):
        self.output_dir = Path(str(output_dir))
        self.input_dir = Path(str(input_dir))
        self.prefixes = [
            ("geometry", "Geometry"),
            ("image", "Image"),
            ("kd_tree", "KD Tree"),
            ("octree", "Octree"),
            ("point_cloud", "Point Cloud"),
            ("ray_casting", "Ray Casting"),
            ("rgbd", "RGBD Image"),
            ("triangle_mesh", "Triangle Mesh"),
            ("voxel_grid", "Voxel Grid"),
        ]

        sys.path.append(os.path.join(pwd, "..", "python", "tools"))
        from cli import _get_all_examples_dict
        self.get_all_examples_dict = _get_all_examples_dict

    def _get_examples_dict(self):
        examples_dict = self.get_all_examples_dict()
        categories_to_remove = [
            "benchmark", "reconstruction_system", "t_reconstruction_system"
        ]
        for cat in categories_to_remove:
            examples_dict.pop(cat, None)  # Use pop with None to avoid KeyError
        return examples_dict

    def _get_prefix(self, example_name):
        for prefix, sub_category in self.prefixes:
            if example_name.startswith(prefix):
                return prefix
        raise Exception("No prefix found for geometry examples")

    @staticmethod
    def _generate_index(title, output_path):
        os.makedirs(output_path, exist_ok=True)
        out_string = (f"{title}\n"
                      f"{'-' * len(title)}\n\n")
        with open(output_path / "index.rst", "w") as f:
            f.write(out_string)

    @staticmethod
    def _add_example_to_docs(example: Path, output_path):
        shutil.copy(example, output_path)
        out_string = (f"{example.name}"
                      f"\n{'`' * (len(example.name))}\n"
                      f"\n.. literalinclude:: {example.name}"
                      f"\n   :language: python"
                      f"\n   :linenos:"
                      f"\n\n\n")

        with open(output_path / "index.rst", "a") as f:
            f.write(out_string)

    def generate_rst(self):
        print(f"Generating *.rst Python example docs in directory: "
              f"{self.output_dir}")
        _create_or_clear_dir(self.output_dir)
        examples_dict = self._get_examples_dict()

        categories = [cat for cat in self.input_dir.iterdir() if cat.is_dir()]

        for cat in categories:
            if cat.stem in examples_dict.keys():
                out_dir = self.output_dir / cat.stem
                if (cat.stem == "geometry"):
                    self._generate_index(cat.stem.capitalize(), out_dir)
                    with open(out_dir / "index.rst", "a") as f:
                        f.write(f".. toctree::\n"
                                f"    :maxdepth: 2\n\n")
                        for prefix, sub_cat in self.prefixes:
                            f.write(f"    {prefix}/index\n")

                    for prefix, sub_category in self.prefixes:
                        self._generate_index(sub_category, out_dir / prefix)
                    examples = sorted(Path(cat).glob("*.py"))
                    for ex in examples:
                        if ex.stem in examples_dict[cat.stem]:
                            prefix = self._get_prefix(ex.stem)
                            sub_category_path = out_dir / prefix
                            self._add_example_to_docs(ex, sub_category_path)
                else:
                    if (cat.stem == "io"):
                        self._generate_index("IO", out_dir)
                    else:
                        self._generate_index(cat.stem.capitalize(), out_dir)

                    examples = sorted(Path(cat).glob("*.py"))
                    for ex in examples:
                        if ex.stem in examples_dict[cat.stem]:
                            shutil.copy(ex, out_dir)
                            self._add_example_to_docs(ex, out_dir)


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
        # Resolve path relative to script directory for robustness
        script_dir = os.path.dirname(os.path.realpath(__file__))
        documented_modules_path = os.path.join(script_dir, "documented_modules.txt")
        with open(documented_modules_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                print(line, end="")
                m = re.match(r"^(cloudViewer\..*)\s*$", line)
                if m:
                    module_names.append(m.group(1))
        print("Documented modules:")
        for module_name in module_names:
            print("-", module_name)
        return module_names

    def _try_import_module(self, full_module_name):
        """Returns the module object for the given module path"""
        try:
            import cloudViewer
            if hasattr(cloudViewer, '_build_config'):
                if cloudViewer._build_config.get('BUILD_TENSORFLOW_OPS', False):
                    import cloudViewer.ml.tf
                if cloudViewer._build_config.get('BUILD_PYTORCH_OPS', False):
                    import cloudViewer.ml.torch
        except (ImportError, AttributeError, KeyError) as e:
            # Optional modules, ignore if not available
            print(f"⚠️  {e}")
            pass
        
        # Try to import the specific module
        try:
            # Try to import directly. This will work for pure python submodules
            module = importlib.import_module(full_module_name)
            return module
        except ImportError:
            # Traverse the module hierarchy of the root module.
            # This code path is necessary for modules for which we manually
            # define a specific module path (e.g. the modules defined with
            # pybind).
            try:
                current_module = cloudViewer
                for sub_module_name in full_module_name.split(".")[1:]:
                    current_module = getattr(current_module, sub_module_name)
                return current_module
            except AttributeError as e:
                raise ImportError(f"Could not find module {full_module_name}: {e}")

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
        if not (full_module_name.startswith("cloudViewer.ml.tf") or
                full_module_name.startswith("cloudViewer.ml.torch")):
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

    def overwrite_tutorial_file(self, url, output_file, output_file_path):
        with urllib.request.urlopen(
                url,
                context=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            with open(output_file, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        shutil.move(output_file, output_file_path)
        
    def run(self):
        """Copy Jupyter notebooks to tutorial directories."""
        print("📓 Copying Jupyter notebooks to tutorial directories...")
        
        # Setting os.environ["CI"] will disable interactive (blocking) mode in
        # Jupyter notebooks
        os.environ["CI"] = "true"

        # Copy from jupyter to the tutorial folder.
        nb_paths = []
        nb_parent_src = Path(self.current_file_dir) / "jupyter"
        nb_parent_dst = Path(self.current_file_dir) / "source" / "tutorial"
        if not nb_parent_src.exists():
            print(f"⚠️  Jupyter directory not found: {nb_parent_src}")
            return
        
        example_dirs = [
            name for name in os.listdir(nb_parent_src)
            if os.path.isdir(nb_parent_src / name)
        ]

        print(f"Copying {nb_parent_src / 'cloudViewer_tutorial.py'} "
              f"to {nb_parent_dst / 'cloudViewer_tutorial.py'}")
        shutil.copy(
            nb_parent_src / "cloudViewer_tutorial.py",
            nb_parent_dst / "cloudViewer_tutorial.py",
        )

        for example_dir in example_dirs:
            in_dir = nb_parent_src / example_dir
            out_dir = nb_parent_dst / example_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            for nb_out_path in out_dir.glob("*.ipynb"):
                print("Delete: {}".format(nb_out_path))
                nb_out_path.unlink()

            for nb_in_path in in_dir.glob("*.ipynb"):
                nb_out_path = out_dir / nb_in_path.name
                _update_file(nb_in_path, nb_out_path)
                nb_paths.append(nb_out_path)

            # Copy the 'images' dir present in some example dirs.
            if (in_dir / "images").is_dir():
                if (out_dir / "images").exists():
                    shutil.rmtree(out_dir / "images")
                print("Copy: {}\n   -> {}".format(in_dir / "images",
                                                  out_dir / "images"))
                shutil.copytree(in_dir / "images", out_dir / "images")

        # Execute Jupyter notebooks
        # Files that should not be executed.
        nb_direct_copy = [
            'draw_plotly.ipynb',
            'hashmap.ipynb',
            'jupyter_visualization.ipynb',
            't_icp_registration.ipynb',
            'tensor.ipynb',
        ]

        for nb_path in nb_paths:
            if nb_path.name in nb_direct_copy:
                print("[Processing notebook {}, directly copied]".format(
                    nb_path.name))
                continue

            print("[Processing notebook {}]".format(nb_path.name))
            with open(nb_path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # https://github.com/spatialaudio/nbsphinx/blob/master/src/nbsphinx.py
            has_code = any(c.source for c in nb.cells if c.cell_type == "code")
            has_output = any(
                c.get("outputs") or c.get("execution_count")
                for c in nb.cells
                if c.cell_type == "code")
            execute = (has_code and not has_output)
            print("has_code: {}, has_output: {}, execute: {}".format(
                has_code, has_output, execute))

            if execute:
                ep = nbconvert.preprocessors.ExecutePreprocessor(timeout=6000)
                try:
                    ep.preprocess(nb, {"metadata": {"path": nb_path.parent}})
                except nbconvert.preprocessors.execute.CellExecutionError:
                    print("Execution of {} failed, this will cause CI to fail.".
                          format(nb_path.name))
                    if "GITHUB_ACTIONS" in os.environ:
                        raise

                with open(nb_path, "w", encoding="utf-8") as f:
                    nbformat.write(nb, f)

        url = "https://github.com/isl-org/Open3D/files/8243984/t_icp_registration.zip"
        output_file = "t_icp_registration.ipynb"
        output_file_path = nb_parent_dst / "t_pipelines" / output_file
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        self.overwrite_tutorial_file(url, output_file, str(output_file_path))
        
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
        
        # Copy docs files from Open3D-ML repo
        cloudviewer_root = os.environ.get(
            "CLOUDVIEWER_ML_ROOT",
            os.path.join(self.current_file_dir, "../../CloudViewer-ML"))
        cloudviewer_docs = [
            os.path.join(cloudviewer_root, "docs", "tensorboard.md")
        ]
        for cloudviewer_doc in cloudviewer_docs:
            if os.path.isfile(cloudviewer_doc):
                shutil.copy(cloudviewer_doc, self.current_file_dir)
        
        build_dir = os.path.join(self.html_output_dir, "html")
        source_dir = os.path.join(self.current_file_dir, "source")
        nproc = multiprocessing.cpu_count() if self.parallel else 1
        print(f"Building docs with {nproc} processes")

        today = os.environ.get("SPHINX_TODAY", None)
        if today:
            cmd_args_today = ["-D", "today=" + today]
        else:
            cmd_args_today = []

        if self.is_release:
            version_list = [
                line.rstrip("\n").split(" ")[1]
                for line in open("../libs/cloudViewer/version.txt")
            ]
            release_version = ".".join(version_list[:3])
            print("Building docs for release:", release_version)

            cmd = [
                "sphinx-build", "-j",
                str(nproc), "-b", "html", "-D", "version=" + release_version,
                "-D", "release=" + release_version
            ] + cmd_args_today + [
                source_dir,
                build_dir,
            ]
        else:
            cmd = [
                "sphinx-build",
                "-j",
                str(nproc),
                "-b",
                "html",
            ] + cmd_args_today + [
                source_dir,
                build_dir,
            ]

        print('Calling: "%s"' % " ".join(cmd))
        
        try:
            subprocess.check_call(cmd, stdout=sys.stdout, stderr=sys.stderr)
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

    # Generate Python example docs BEFORE Sphinx build
    if args.sphinx:
        py_example_input_dir = os.path.join(pwd, "..", "examples", "Python")
        if os.path.exists(py_example_input_dir):
            print("Building Python example reST")
            pe = PyExampleDocsBuilder(input_dir=py_example_input_dir, pwd=pwd)
            pe.generate_rst()
        else:
            print(f"⚠️  Python examples directory not found: {py_example_input_dir}")

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

