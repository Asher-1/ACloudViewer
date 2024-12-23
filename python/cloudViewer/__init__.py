# ----------------------------------------------------------------------------
# -                        CloudViewer: asher-1.github.io                    -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 asher-1.github.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

# Workaround when multiple copies of the OpenMP runtime have been linked to
# the program, which happens when PyTorch loads OpenMP runtime first. Not that
# this method is "unsafe, unsupported, undocumented", but we found it to be
# generally safe to use. This should be deprecated once we found a way to
# "ensure that only a single OpenMP runtime is linked into the process".
#
# https://github.com/llvm-mirror/openmp/blob/8453ca8594e1a5dd8a250c39bf8fcfbfb1760e60/runtime/src/i18n/en_US.txt#L449
# https://github.com/dmlc/xgboost/issues/1715

import os
import re
import sys
import platform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Enable thread composability manager to coordinate Intel OpenMP and TBB threads. Only works with Intel OpenMP.
# TBB must not be already loaded.
os.environ["TCM_ENABLE"] = "1"
from ctypes import CDLL
from ctypes.util import find_library
from pathlib import Path
import warnings
from cloudViewer._build_config import _build_config

MAIN_LIB_PATH=Path(__file__).parent / "lib"

def load_cdll(path):
    """
    Wrapper around ctypes.CDLL to take care of Windows compatibility.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Shared library file not found: {path}.")

    if sys.platform == "win32" and sys.version_info >= (3, 8):
        # https://stackoverflow.com/a/64472088/1255535
        return CDLL(str(path), winmode=0)
    else:
        return CDLL(str(path))

def try_load_cdll(so_name):
    """
    Wrapper around ctypes.CDLL to take care of Linux compatibility.
    """
    try:  # StopIteration if lib not available
        libs_file_list = list(MAIN_LIB_PATH.glob(so_name))
        if len(libs_file_list) > 0:
            if len(libs_file_list) > 1:
                warnings.warn(
                    f"cloudViewer Found multiple libs named: {libs_file_list}",
                    ImportWarning,)
            load_cdll(str(next(MAIN_LIB_PATH.glob(so_name))))
    except OSError as os_error:
        match = re.search(r'lib[^/]+\.so[^\s:]*', str(os_error))
        missing_so_name = match.group(0) if match else ''
        warnings.warn(
            f"{os_error} when loading {libs_file_list}, maybe you should load {missing_so_name} first",
            ImportWarning,)
    except StopIteration as e:  # so_name not available
        warnings.warn(
            f"{e} \nFailed to load {libs_file_list}.",
            ImportWarning,
        )

if _build_config["BUILD_GUI"] and not (find_library('c++abi') or
                                       find_library('c++')):
    try:  # Preload libc++.so and libc++abi.so (required by filament)
        load_cdll(str(next((Path(__file__).parent).glob('*c++abi.*'))))
        load_cdll(str(next((Path(__file__).parent).glob('*c++.*'))))
    except StopIteration:  # Not found: check system paths while loading
        pass

# fix link bugs for qt on Linux platform
if os.path.exists(MAIN_LIB_PATH):
    LIB_PATH = str(MAIN_LIB_PATH)
    # the Qt plugin is included currently only in the pre-built wheels
    if _build_config["BUILD_RECONSTRUCTION"]:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = LIB_PATH

    if platform.system() == "Windows":
        os.environ['path'] = LIB_PATH + ";" + os.environ['path']
    else:
        os.environ['PATH'] = LIB_PATH + ":" + os.environ['PATH']

    if platform.system() == "Linux":  # must load shared library in order on linux
        # os.environ['LD_LIBRARY_PATH'] = LIB_PATH + ":" +  os.environ.get('LD_LIBRARY_PATH', '')
        os.environ['LD_LIBRARY_PATH'] = LIB_PATH
        try_load_cdll('libtbb*') # fix missing libtbb.so
        try_load_cdll('libicudata*')
        try_load_cdll('libicuuc*')
        try_load_cdll('libicui18n*')
        try_load_cdll('libQt5Core*')
        try_load_cdll('libQt5Gui*')
        try_load_cdll('libQt5Widgets*')
        try_load_cdll('libQt5Concurrent*')

        try_load_cdll('libCVCoreLib*')
        try_load_cdll('libECV_DB_LIB*')
        try_load_cdll('libECV_IO_LIB*')
        if _build_config["BUILD_RECONSTRUCTION"]:
            try_load_cdll('libboost_program_options*')
            try_load_cdll('libboost_filesystem*')
            try_load_cdll('libboost_system*')
            try_load_cdll('libboost_iostreams*')
            try_load_cdll('libQt5DBus*')
            try_load_cdll('libQt5XcbQpa*')
            try_load_cdll('libfreeimage*')
            try_load_cdll('libgflags*')
            try_load_cdll('libglog*')
            try_load_cdll('libceres*')
        # no need?
        load_cdll(str(next(MAIN_LIB_PATH.glob('lib*'))))


__DEVICE_API__ = 'cpu'
if _build_config["BUILD_CUDA_MODULE"]:
    # Load CPU pybind dll gracefully without introducing new python variable.
    # Do this before loading the CUDA pybind dll to correctly resolve symbols
    try:  # StopIteration if cpu version not available
        load_cdll(str(next((Path(__file__).parent / 'cpu').glob('pybind*'))))
    except StopIteration:
        warnings.warn(
        "cloudViewer was built with CUDA support, but cloudViewer CPU Python "
        "bindings were not found. cloudViewer will not work on systems without"
        " CUDA devices.",
        ImportWarning,)
    try:
        # Check CUDA availability without importing CUDA pybind symbols to
        # prevent "symbol already registered" errors if first import fails.
        _pybind_cuda = load_cdll(str(next((Path(__file__).parent / 'cuda').glob('pybind*'))))
        if _pybind_cuda.cloudViewer_core_cuda_device_count() > 0:
            from cloudViewer.cuda.pybind import (
                core,
                camera,
                geometry,
                io,
                pipelines,
                utility,
                t,
            )
            from cloudViewer.cuda import pybind
            
            if _build_config["BUILD_RECONSTRUCTION"]:
                from cloudViewer.cuda.pybind import reconstruction
                
            __DEVICE_API__ = 'cuda'
        else:
            warnings.warn(
                "cloudViewer was built with CUDA support, but no suitable CUDA "
                "devices found. If your system has CUDA devices, check your "
                "CUDA drivers and runtime.",
                ImportWarning,
            )
    except OSError as os_error:
        warnings.warn(
            f"cloudViewer was built with CUDA support, but an error ocurred while loading the cloudViewer CUDA Python bindings. This is usually because the CUDA libraries could not be found. Check your CUDA installation. Falling back to the CPU pybind library. Reported error: {os_error}.",
            ImportWarning,
        )
    except StopIteration as e:  # pybind cuda library not available
        print(e)
        warnings.warn(
            "cloudViewer was built with CUDA support, but cloudViewer CUDA Python "
            "binding library not found! Falling back to the CPU Python "
            "binding library.",
            ImportWarning,
        )


if __DEVICE_API__ == 'cpu':
    if sys.platform == "win32":
        try:  # StopIteration if cpu version not available
            load_cdll(str(next((Path(__file__).parent / 'cpu').glob('pybind*'))))
        except StopIteration:
            warnings.warn(
                "cloudViewer CPU Python bindings were not found. cloudViewer will not work on systems.",
                ImportWarning, )
        except Exception as e:
            warnings.warn(str(e))
            
    try:
        from cloudViewer.cpu.pybind import (
            core,
            camera,
            geometry,
            io,
            pipelines,
            utility,
            t,
        )
        from cloudViewer.cpu import pybind

        if _build_config["BUILD_RECONSTRUCTION"]:
            from cloudViewer.cpu.pybind import reconstruction
    except OSError as os_error:
        warnings.warn(
            f"cloudViewer was built with CPU support. Reported os error: {os_error}.",
            ImportWarning,
        )
    except Exception as e:
        warnings.warn(str(e))


def _insert_pybind_names(skip_names=()):
    """Introduce pybind names as cloudViewer names. Skip names corresponding to
    python subpackages, since they have a different import mechanism."""
    submodules = {}
    for modname in sys.modules:
        if "cloudViewer." + __DEVICE_API__ + ".pybind" in modname:
            if any("." + skip_name in modname for skip_name in skip_names):
                continue
            subname = modname.replace(__DEVICE_API__ + ".pybind.", "")
            if subname not in sys.modules:
                submodules[subname] = sys.modules[modname]
    sys.modules.update(submodules)

import cloudViewer.visualization

_insert_pybind_names(skip_names=("ml",))

__version__ = "@PROJECT_VERSION@"

if int(sys.version_info[0]) < 3:
    raise Exception("CloudViewer only supports Python 3.")

if _build_config["BUILD_JUPYTER_EXTENSION"]:
    import platform

    if not (platform.machine().startswith("arm") or
            platform.machine().startswith("aarch")):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                print("Jupyter environment detected. "
                      "Enabling CloudViewer WebVisualizer.")
                # Set default window system.
                cloudViewer.visualization.webrtc_server.enable_webrtc()
                # HTTP handshake server is needed when CloudViewer is serving the
                # visualizer webpage. Disable since Jupyter is serving.
                cloudViewer.visualization.webrtc_server.disable_http_handshake()
        except NameError:
            pass
    else:
        warnings.warn("cloudViewer WebVisualizer is not supported on ARM for now.",
                RuntimeWarning)

# CLOUDVIEWER_ML_ROOT points to the root of the CloudViewer-ML repo.
# If set this will override the integrated CloudViewer-ML.
if 'CLOUDVIEWER_ML_ROOT' in os.environ:
    print('Using external CloudViewer-ML in {}'.format(os.environ['CLOUDVIEWER_ML_ROOT']))
    sys.path.append(os.environ['CLOUDVIEWER_ML_ROOT'])
import cloudViewer.ml

# Finally insert pybind names corresponding to ml
_insert_pybind_names()


def _jupyter_labextension_paths():
    """Called by Jupyter Lab Server to detect if it is a valid labextension and
    to install the widget.

    Returns:
        src: Source directory name to copy files from. Webpack outputs generated
            files into this directory and Jupyter Lab copies from this directory
            during widget installation.
        dest: Destination directory name to install widget files to. Jupyter Lab
            copies from `src` directory into <jupyter path>/labextensions/<dest>
            directory during widget installation.
    """
    return [{
        'src': 'labextension',
        'dest': 'cloudViewer',
    }]


def _jupyter_nbextension_paths():
    """Called by Jupyter Notebook Server to detect if it is a valid nbextension
    and to install the widget.

    Returns:
        section: The section of the Jupyter Notebook Server to change.
            Must be 'notebook' for widget extensions.
        src: Source directory name to copy files from. Webpack outputs generated
            files into this directory and Jupyter Notebook copies from this
            directory during widget installation.
        dest: Destination directory name to install widget files to. Jupyter
            Notebook copies from `src` directory into
            <jupyter path>/nbextensions/<dest> directory during widget
            installation.
        require: Path to importable AMD Javascript module inside the
            <jupyter path>/nbextensions/<dest> directory.
    """
    return [{
        'section': 'notebook',
        'src': 'nbextension',
        'dest': 'cloudViewer',
        'require': 'cloudViewer/extension'
    }]
    
del os, re, sys, platform, CDLL, load_cdll, find_library, Path, warnings, _insert_pybind_names