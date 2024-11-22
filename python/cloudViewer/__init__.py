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
import sys
import warnings
import platform

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Enable thread composability manager to coordinate Intel OpenMP and TBB threads. Only works with Intel OpenMP.
# TBB must not be already loaded.
os.environ["TCM_ENABLE"] = "1"
from ctypes import CDLL as _CDLL
from ctypes.util import find_library as _find_library
from pathlib import Path as _Path

from cloudViewer._build_config import _build_config

if _build_config["BUILD_GUI"] and not (_find_library('c++abi') or
                                       _find_library('c++')):
    try:  # Preload libc++.so and libc++abi.so (required by filament)
        _CDLL(str(next((_Path(__file__).parent).glob('*c++abi.*'))))
        _CDLL(str(next((_Path(__file__).parent).glob('*c++.*'))))
    except StopIteration:  # Not found: check system paths while loading
        pass

# fix link bugs for qt on Linux platform
if os.path.exists(_Path(__file__).parent / 'libs'):
    LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libs")
    # the Qt plugin is included currently only in the pre-built wheels
    if _build_config["BUILD_RECONSTRUCTION"]:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = LIB_PATH

    if platform.system() == "Windows":
        os.environ['path'] = LIB_PATH + ";" + os.environ['path']
    else:
        os.environ['PATH'] = LIB_PATH + ";" + os.environ['PATH']

    if platform.system() == "Linux":  # must load shared library in order on linux
        try:  # Try loading libstdc++* if possible to fix cannot found GLIB_XX issues
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('*stdc++*'))))
        except StopIteration:
            pass
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libicudata*'))))
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libicuuc*'))))
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libicui18n*'))))

        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libQt5Core*'))))
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libQt5Gui*'))))
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libQt5Widgets*'))))
        libqts = (_Path(__file__).parent / 'libs').glob('libQt5*')
        libqts = sorted(libqts)
        for lib in libqts:
            _CDLL(lib)

        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libCVCoreLib*'))))
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libECV_DB_LIB*'))))
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libECV_IO_LIB*'))))
        if _build_config["BUILD_RECONSTRUCTION"]:
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libfreeimage*'))))
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libgflags*'))))
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libglog*'))))
            # _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libgfortran*'))))
            # _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libblas*'))))
            # _CDLL(str(next((_Path(__file__).parent / 'libs').glob('liblapack*'))))
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libceres*'))))
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libboost_program_options*'))))
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libboost_filesystem*'))))
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libboost_system*'))))
            _CDLL(str(next((_Path(__file__).parent / 'libs').glob('libboost_iostreams*'))))
        _CDLL(str(next((_Path(__file__).parent / 'libs').glob('lib*'))))

__DEVICE_API__ = 'cpu'
if _build_config["BUILD_CUDA_MODULE"]:
    # fix search mechanism
    if platform.system() == "Windows" and sys.version_info >= (3, 8):
        __DEVICE_API__ = 'cuda'
    else:
        # Load CPU pybind dll gracefully without introducing new python variable.
        # Do this before loading the CUDA pybind dll to correctly resolve symbols
        try:  # StopIteration if cpu version not available
            _CDLL(str(next((_Path(__file__).parent / 'cpu').glob('pybind*'))))
        except StopIteration:
            warnings.warn(
            "cloudViewer was built with CUDA support, but cloudViewer CPU Python "
            "bindings were not found. cloudViewer will not work on systems without"
            " CUDA devices.",
            ImportWarning,)
        try:
            # Check CUDA availability without importing CUDA pybind symbols to
            # prevent "symbol already registered" errors if first import fails.
            _pybind_cuda = _CDLL(str(next((_Path(__file__).parent / 'cuda').glob('pybind*'))))
            if _pybind_cuda.cloudViewer_core_cuda_device_count() > 0:
                __DEVICE_API__ = 'cuda'
        except OSError as e:  # CUDA not installed
            print(e)
            pass
        except StopIteration as e:  # pybind cuda library not available
            print(e)
            pass

if __DEVICE_API__ == 'cpu':
    from cloudViewer.cpu.pybind import (camera, geometry, io, pipelines, utility, t)
    from cloudViewer.cpu import pybind

    if _build_config["BUILD_RECONSTRUCTION"]:
        from cloudViewer.cpu.pybind import reconstruction
elif __DEVICE_API__ == "cuda":
    from cloudViewer.cuda.pybind import (camera, geometry, io, pipelines, utility, t)
    from cloudViewer.cuda import pybind

    if _build_config["BUILD_RECONSTRUCTION"]:
        from cloudViewer.cuda.pybind import reconstruction
else:
    print("[WARNING] Unsupported device api: " + __DEVICE_API__)

import cloudViewer.core
import cloudViewer.visualization

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
        print("CloudViewer WebVisualizer is not supported on ARM for now.")
        pass

# CLOUDVIEWER_ML_ROOT points to the root of the CloudViewer-ML repo.
# If set this will override the integrated CloudViewer-ML.
if 'CLOUDVIEWER_ML_ROOT' in os.environ:
    print('Using external CloudViewer-ML in {}'.format(os.environ['CLOUDVIEWER_ML_ROOT']))
    sys.path.append(os.environ['CLOUDVIEWER_ML_ROOT'])
import cloudViewer.ml


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
