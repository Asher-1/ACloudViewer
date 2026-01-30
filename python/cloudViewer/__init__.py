# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Enable thread composability manager to coordinate Intel OpenMP and TBB threads. Only works with Intel OpenMP.
# TBB must not be already loaded.
os.environ["TCM_ENABLE"] = "1"
from ctypes import CDLL
from ctypes.util import find_library
from pathlib import Path
import warnings
from cloudViewer._build_config import _build_config

MAIN_LIB_PATH = Path(__file__).parent / "lib"

if sys.platform == "win32":  # Unix: Use rpath to find libraries
    _win32_dll_dir = os.add_dll_directory(str(Path(__file__).parent))

# Cache for loaded libraries to prevent duplicate loading
_loaded_libs_cache = {}


def load_cdll(path):
    """
    Wrapper around ctypes.CDLL to take care of Windows compatibility.
    Prevents duplicate loading by caching loaded libraries.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Shared library file not found: {file_path}.")

    # Use absolute path as cache key to handle relative/absolute path variations
    abs_path = str(file_path.resolve())

    # Return cached library if already loaded
    if abs_path in _loaded_libs_cache:
        return _loaded_libs_cache[abs_path]

    # Load the library
    if sys.platform == "win32" and sys.version_info >= (3, 8):
        # https://stackoverflow.com/a/64472088/1255535
        lib = CDLL(str(file_path), winmode=0)
    else:
        lib = CDLL(str(file_path))

    # Cache the loaded library
    _loaded_libs_cache[abs_path] = lib
    return lib


def try_load_cdll(so_name, max_retries=3):
    """
    Wrapper around ctypes.CDLL to take care of Linux compatibility.
    Automatically detects and loads missing dependencies.
    
    Args:
        so_name: Library name pattern to load (e.g., 'libQt5Core*')
        max_retries: Maximum number of retry attempts to resolve dependencies
    """
    try:  # StopIteration if lib not available
        libs_file_list = list(MAIN_LIB_PATH.glob(so_name))
        if len(libs_file_list) > 0:
            if len(libs_file_list) > 1:
                warnings.warn(
                    f"cloudViewer Found multiple libs named: {libs_file_list}",
                    ImportWarning,
                )
            load_cdll(str(next(MAIN_LIB_PATH.glob(so_name))))
    except OSError as os_error:
        # Try to extract the missing library name from the error message
        match = re.search(r'lib[^/]+\.so[^\s:]*', str(os_error))
        missing_so_name = match.group(0) if match else ''

        if missing_so_name and max_retries > 0:
            # Try to find and load the missing dependency
            missing_pattern = missing_so_name.replace('.so', '*')
            missing_libs = list(MAIN_LIB_PATH.glob(missing_pattern))

            if missing_libs:
                # Found the missing dependency, try to load it first
                try:
                    load_cdll(str(missing_libs[0]))
                    # Retry loading the original library
                    try_load_cdll(so_name, max_retries - 1)
                    return  # Success
                except Exception:
                    pass  # Fall through to warning

        warnings.warn(
            f"{os_error} when loading {libs_file_list}, maybe you should load {missing_so_name} first",
            ImportWarning,
        )
    except StopIteration as e:  # so_name not available
        warnings.warn(
            f"{e} \nFailed to load {libs_file_list}.",
            ImportWarning,
        )


# Qt version detection and loading utilities
_qt_version_cache = None


def _detect_qt_version():
    """
    Detect available Qt version (Qt6 or Qt5).
    Caches the result to avoid repeated detection.
    
    Returns:
        int: Qt version (6, 5) or None if neither is found
    """
    global _qt_version_cache
    if _qt_version_cache is not None:
        return _qt_version_cache

    qt6_core = list(MAIN_LIB_PATH.glob('libQt6Core*'))
    qt5_core = list(MAIN_LIB_PATH.glob('libQt5Core*'))

    if qt6_core:
        _qt_version_cache = 6
    elif qt5_core:
        _qt_version_cache = 5
    else:
        _qt_version_cache = None

    return _qt_version_cache


def _load_qt_library(lib_name, required=True):
    """
    Load a Qt library based on the detected Qt version.
    
    Args:
        lib_name: Base library name without version prefix (e.g., 'Core', 'Gui', 'Widgets')
        required: If True, warn if library cannot be loaded. If False, silently skip.
    
    Examples:
        _load_qt_library('Core')  # Loads libQt6Core* or libQt5Core*
        _load_qt_library('DBus')  # Loads libQt6DBus* or libQt5DBus*
    """
    detected_version = _detect_qt_version()

    if detected_version == 6:
        pattern = f'libQt6{lib_name}*'
    elif detected_version == 5:
        pattern = f'libQt5{lib_name}*'
    else:
        if required:
            warnings.warn(
                f"Could not detect Qt version, attempting to load Qt5 library: libQt5{lib_name}*",
                ImportWarning,
            )
        pattern = f'libQt5{lib_name}*'

    try_load_cdll(pattern)


def _load_qt_libraries(lib_names, required=True):
    """
    Load multiple Qt libraries based on the detected Qt version.
    
    Args:
        lib_names: List of base library names without version prefix
        required: If True, warn if libraries cannot be loaded. If False, silently skip.
    
    Examples:
        _load_qt_libraries(['Core', 'Gui', 'Widgets'])
    """
    for lib_name in lib_names:
        _load_qt_library(lib_name, required)


# Platform-specific environment setup utilities
def _setup_path(lib_path,
                path_separator=':',
                env_var='PATH',
                keep_old_path=True):
    """
    Set up PATH environment variable for any platform.
    Filters out conda/system Qt paths to avoid mixing Qt versions.
    
    Args:
        lib_path: Path to the library directory
        path_separator: Path separator character (default: ':' for Unix, ';' for Windows)
        env_var: Environment variable name (default: 'PATH', Windows can use 'path')
        keep_old_path: If True, append old path when filtered_paths is empty (Unix behavior).
                       If False, only set lib_path when filtered_paths is empty (Windows behavior).
    """
    # Windows: try 'path' first, then fallback to 'PATH' (case-insensitive)
    if sys.platform == "win32":
        old_path = os.environ.get('path', '') or os.environ.get('PATH', '')
    else:
        old_path = os.environ.get(env_var, '')

    # Filter out conda/system Qt paths to avoid mixing Qt versions
    filtered_paths = [
        p for p in old_path.split(path_separator)
        if p and not any(x in p.lower() for x in [
            'qt5', 'qt\\5', 'qt-5', 'qt6', 'qt\\6', 'qt-6', 'anaconda',
            'miniconda', 'conda'
        ])
    ]

    if filtered_paths:
        os.environ[env_var] = lib_path + path_separator + path_separator.join(
            filtered_paths)
    else:
        if keep_old_path:
            os.environ[env_var] = lib_path + path_separator + old_path
        else:
            os.environ[env_var] = lib_path


def _setup_qt_plugin_path(lib_path):
    """
    Set up Qt plugin path environment variables for Windows and macOS.
    
    Args:
        lib_path: Path to the library directory
    """
    qt_plugin_path = os.path.join(lib_path, 'plugins')
    if os.path.exists(qt_plugin_path):
        os.environ['QT_PLUGIN_PATH'] = qt_plugin_path
    else:
        os.environ['QT_PLUGIN_PATH'] = lib_path

    # Set platform plugin path - use platforms subdirectory if it exists
    platform_plugin_path = os.path.join(lib_path, 'platforms')
    if os.path.exists(platform_plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platform_plugin_path
    else:
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = lib_path


def _filter_qt_paths(paths, path_separator=':'):
    """
    Filter out system Qt paths from a list of paths.
    
    Args:
        paths: List of paths or path string (will be split by path_separator)
        path_separator: Path separator character (default: ':' for Unix, ';' for Windows)
    
    Returns:
        List of filtered paths (system Qt paths removed)
    """
    if isinstance(paths, str):
        path_list = paths.split(path_separator) if paths else []
    else:
        path_list = paths

    # Common Qt-related keywords to filter
    qt_keywords = [
        'qt5',
        'qt-5',
        'qt6',
        'qt-6',
        'qt/',
        'qt\\',
        'anaconda',
        'miniconda',
        'conda',
    ]

    # Platform-specific Qt paths
    if sys.platform == "linux":
        qt_keywords.extend([
            '/usr/lib/x86_64-linux-gnu/qt',
            '/usr/lib/qt',
            '/usr/local/qt',
        ])
    elif sys.platform == "darwin":
        qt_keywords.extend([
            '/usr/local/opt/qt',
            '/opt/homebrew/opt/qt',
            '/opt/local/lib',
        ])
    elif sys.platform == "win32":
        qt_keywords.extend([
            'c:\\qt',
            'c:\\program files\\qt',
            'c:\\program files (x86)\\qt',
        ])

    # Qt library/framework identifiers
    qt_lib_identifiers = [
        'libQt5', 'libQt6', 'Qt5Core', 'Qt6Core', 'Qt5Core.framework',
        'Qt6Core.framework', 'QtCore.framework', 'qt5core', 'qt6core', 'qtcore',
        'qtwidgets', 'qt5gui', 'qt6gui', 'qtgui', 'qt5', 'qt6'
    ]

    filtered_paths = [
        p for p in path_list
        if p and not any(x in p.lower() for x in qt_keywords) and not (any(
            qt_id in p for qt_id in qt_lib_identifiers))
    ]

    return filtered_paths


def _setup_library_paths_early(lib_path):
    """
    Set up library search paths for all platforms BEFORE loading any libraries.
    This prevents mixing system Qt libraries with package Qt libraries.
    
    Args:
        lib_path: Path to the package library directory
    """
    if sys.platform == "linux":
        # Linux: Set up LD_LIBRARY_PATH
        old_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        filtered_paths = _filter_qt_paths(old_ld_path, ':')

        if filtered_paths:
            os.environ['LD_LIBRARY_PATH'] = lib_path + ":" + ":".join(
                filtered_paths)
        else:
            os.environ['LD_LIBRARY_PATH'] = lib_path

    elif sys.platform == "darwin":
        # macOS: Set up DYLD_LIBRARY_PATH and DYLD_FRAMEWORK_PATH
        old_dyld_lib_path = os.environ.get('DYLD_LIBRARY_PATH', '')
        old_dyld_fw_path = os.environ.get('DYLD_FRAMEWORK_PATH', '')

        # Filter library paths
        filtered_lib_paths = _filter_qt_paths(old_dyld_lib_path, ':')
        if filtered_lib_paths:
            os.environ['DYLD_LIBRARY_PATH'] = lib_path + ":" + ":".join(
                filtered_lib_paths)
        else:
            os.environ['DYLD_LIBRARY_PATH'] = lib_path

        # Filter framework paths
        filtered_fw_paths = _filter_qt_paths(old_dyld_fw_path, ':')
        if filtered_fw_paths:
            os.environ['DYLD_FRAMEWORK_PATH'] = lib_path + ":" + ":".join(
                filtered_fw_paths)
        else:
            os.environ['DYLD_FRAMEWORK_PATH'] = lib_path

    elif sys.platform == "win32":
        # Windows: Set up PATH
        old_path = os.environ.get('path', '') or os.environ.get('PATH', '')
        filtered_paths = _filter_qt_paths(old_path, ';')

        # Convert lib_path to Windows format if needed
        lib_path_win = lib_path.replace('/',
                                        '\\') if '/' in lib_path else lib_path

        if filtered_paths:
            os.environ['PATH'] = lib_path_win + ";" + ";".join(filtered_paths)
            os.environ['path'] = os.environ[
                'PATH']  # Also set lowercase version
        else:
            os.environ['PATH'] = lib_path_win
            os.environ['path'] = lib_path_win


# Linux-specific library loading utilities
def _setup_linux_libraries():
    """
    Set up environment variables and load all Linux-specific libraries.
    Must be called in the correct order to resolve dependencies.
    """
    # 1. Set up environment variables
    # NOTE: LD_LIBRARY_PATH should already be set earlier in the module initialization
    # (around line 403-428) to prevent loading system Qt libraries.
    # Here we just verify it's set correctly - no need to set it again.
    # If for some reason it wasn't set (e.g., non-standard import order),
    # we'll use the current value which should already be filtered.

    # Set Qt-specific environment variables to avoid mixing system Qt
    # This ensures all Qt components come from the same version
    linux_qt_plugin_path = os.path.join(LIB_PATH, 'plugins')
    if os.path.exists(linux_qt_plugin_path):
        os.environ['QT_PLUGIN_PATH'] = linux_qt_plugin_path
    else:
        # Fallback to lib directory if plugins subdirectory doesn't exist
        os.environ['QT_PLUGIN_PATH'] = LIB_PATH

    # Set platform plugin path for both possible locations
    platform_plugin_path = os.path.join(LIB_PATH, 'platforms')
    if os.path.exists(platform_plugin_path):
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = platform_plugin_path

    # 2. Load base dependencies (tbb, icu libraries, double-conversion, md4c)
    try_load_cdll('libtbb*')  # fix missing libtbb.so
    try_load_cdll('libicudata*')
    try_load_cdll('libicuuc*')
    try_load_cdll('libicui18n*')
    if len(list(MAIN_LIB_PATH.glob('libdouble-conversion*'))) > 0:
        try_load_cdll('libdouble-conversion*')
    if len(list(MAIN_LIB_PATH.glob('libmd4c*'))) > 0:
        try_load_cdll('libmd4c*')

    # 3. Load Qt core libraries and Qt6-specific OpenGL libraries
    qt_version = _detect_qt_version()
    # Core Qt libraries
    core_libs = ['Core', 'Gui', 'Widgets', 'Concurrent', 'Svg']
    _load_qt_libraries(core_libs)
    # Qt6-specific OpenGL libraries (Qt5 uses OpenGL from Widgets)
    if qt_version == 6:
        _load_qt_library('OpenGL', required=False)
        _load_qt_library('OpenGLWidgets', required=False)

    # 4. Load X11/XCB libraries (must be before QtXcbQpa)
    # fix symbol lookup error: libQt5XcbQpa.so.5 / libQt6XcbQpa.so.6:
    # maybe you should load libxcb-icccm.so.4 first
    # Load X11/XCB related libraries in dependency order before QtXcbQpa
    # These libraries must be loaded in the correct order to resolve all dependencies
    x11_xcb_libs = [
        # Base X11 libraries
        'libxcb.so*',  # Base XCB library (must be first)
        'libX11.so*',  # X11 library
        'libX11-xcb*',  # X11-XCB bridge
        'libXext*',  # X11 extensions
        # XKB libraries (needed by QtXcbQpa)
        'libxkbcommon.so*',  # Base XKB library (before x11 variant)
        'libxkbcommon-x11*',  # XKB X11 integration
        # XCB utility libraries
        'libxcb-util*',  # Base utility library
        'libxcb-shm*',  # Shared memory (needed by image)
        'libxcb-icccm*',  # ICCCM depends on util
        'libxcb-image*',  # Image depends on util and shm
        'libxcb-keysyms*',  # Keysyms depends on util
        'libxcb-render-util*',  # Render util
        'libxcb-render.*',  # Render
        'libxcb-shape*',  # Shape extension
        'libxcb-sync*',  # Sync extension
        'libxcb-xfixes*',  # Xfixes extension
        'libxcb-xinerama*',  # Xinerama extension
        'libxcb-xkb*',  # XKB extension
        'libxcb-randr*',  # RandR extension
        'libxcb-xinput*',  # XInput extension
    ]
    for lib in x11_xcb_libs:
        if len(list(MAIN_LIB_PATH.glob(lib))) > 0:
            try_load_cdll(lib)

    # 5. Load project-specific core libraries
    try_load_cdll('libCVCoreLib*')
    try_load_cdll('libCV_DB_LIB*')
    try_load_cdll('libCV_IO_LIB*')

    # 6. Load reconstruction libraries (if enabled)
    if _build_config["BUILD_RECONSTRUCTION"]:
        try_load_cdll('libboost_program_options*')
        try_load_cdll('libboost_filesystem*')
        try_load_cdll('libboost_system*')
        try_load_cdll('libboost_iostreams*')
        # Load Qt DBus and XcbQpa based on detected Qt version
        _load_qt_libraries(['DBus', 'XcbQpa'])
        try_load_cdll('libfreeimage*')
        try_load_cdll('libgflags*')
        try_load_cdll('libglog*')
        try_load_cdll('libatlas*')
        try_load_cdll('libblas*')
        try_load_cdll('liblapack*')
        try_load_cdll('libceres*')

    # 7. Load SSL/crypto libraries (required for libcurl)
    try_load_cdll('libcrypto*')
    try_load_cdll('libssl*')

    # 8. Load remaining libraries
    load_cdll(str(next(MAIN_LIB_PATH.glob('lib*'))))


if _build_config["BUILD_GUI"] and not (find_library('c++abi') or
                                       find_library('c++')):
    try:  # Preload libc++.so and libc++abi.so (required by filament)
        load_cdll(str(next((Path(__file__).parent).glob('*c++abi.*'))))
        load_cdll(str(next((Path(__file__).parent).glob('*c++.*'))))
    except StopIteration:  # Not found: check system paths while loading
        pass

# fix link bugs for qt on Linux platform
# CRITICAL: Set up environment variables BEFORE loading any libraries
# to prevent mixing system Qt libraries with package Qt libraries
if os.path.exists(MAIN_LIB_PATH):
    LIB_PATH = str(MAIN_LIB_PATH)

    # Set up library paths FIRST to prevent loading system Qt libraries
    # This must be done before loading any libraries to avoid version conflicts
    _setup_library_paths_early(LIB_PATH)

    # the Qt plugin is included currently only in the pre-built wheels
    if _build_config["BUILD_RECONSTRUCTION"]:
        # Set platform plugin path - use platforms subdirectory if it exists
        platform_plugin_path = os.path.join(LIB_PATH, 'platforms')
        if os.path.exists(platform_plugin_path):
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = platform_plugin_path
        else:
            os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = LIB_PATH

    if sys.platform == "win32":
        # Windows: Set PATH and Qt plugin paths
        _setup_path(LIB_PATH,
                    path_separator=';',
                    env_var='path',
                    keep_old_path=False)
        _setup_qt_plugin_path(LIB_PATH)
    elif sys.platform == "darwin":
        # macOS: Set PATH and Qt plugin paths
        _setup_path(LIB_PATH)
        _setup_qt_plugin_path(LIB_PATH)
    elif sys.platform == "linux":
        # Linux: Set PATH (QT_PLUGIN_PATH will be set in _setup_linux_libraries)
        _setup_path(LIB_PATH)
        _setup_linux_libraries()

__DEVICE_API__ = 'cpu'
if _build_config["BUILD_CUDA_MODULE"]:
    # Load CPU pybind dll gracefully without introducing new python variable.
    # Do this before loading the CUDA pybind dll to correctly resolve symbols
    try:  # StopIteration if cpu version not available
        load_cdll(str(next((Path(__file__).parent / 'cpu').glob('pybind*'))))
    except StopIteration:
        warnings.warn(
            "CloudViewer was built with CUDA support, but CloudViewer CPU Python "
            "bindings were not found. CloudViewer will not work on systems without"
            " CUDA devices.",
            ImportWarning,
        )
    try:
        if sys.platform == "win32" and sys.version_info >= (3, 8):
            # Since Python 3.8, the PATH environment variable is not used to find DLLs anymore.
            # To allow Windows users to use Open3D with CUDA without running into dependency-problems,
            # look for the CUDA bin directory in PATH and explicitly add it to the DLL search path.
            cuda_bin_path = None
            for path in os.environ['PATH'].split(';'):
                # search heuristic: look for a path containing "cuda" and "bin" in this order.
                if re.search(r'cuda.*bin', path, re.IGNORECASE):
                    cuda_bin_path = path
                    break

            if cuda_bin_path:
                os.add_dll_directory(cuda_bin_path)

        # Check CUDA availability without importing CUDA pybind symbols to
        # prevent "symbol already registered" errors if first import fails.
        _pybind_cuda = load_cdll(
            str(next((Path(__file__).parent / 'cuda').glob('pybind*'))))
        if _pybind_cuda.cloudViewer_core_cuda_device_count() > 0:
            from cloudViewer.cuda.pybind import (
                core,
                camera,
                data,
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
            load_cdll(str(next(
                (Path(__file__).parent / 'cpu').glob('pybind*'))))
        except StopIteration:
            warnings.warn(
                "cloudViewer CPU Python bindings were not found. cloudViewer will not work on systems.",
                ImportWarning,
            )
        except Exception as e:
            warnings.warn(str(e))

    try:
        from cloudViewer.cpu.pybind import (
            core,
            camera,
            data,
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

if (_build_config["BUILD_JUPYTER_EXTENSION"] and os.environ.get(
        "CLOUDVIEWER_DISABLE_WEB_VISUALIZER", "False").lower() != "true"):
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
        warnings.warn(
            "cloudViewer WebVisualizer is not supported on ARM for now.",
            RuntimeWarning)

# CLOUDVIEWER_ML_ROOT points to the root of the CloudViewer-ML repo.
# If set this will override the integrated CloudViewer-ML.
if 'CLOUDVIEWER_ML_ROOT' in os.environ:
    print('Using external CloudViewer-ML in {}'.format(
        os.environ['CLOUDVIEWER_ML_ROOT']))
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


if sys.platform == "win32":
    _win32_dll_dir.close()
del (os, re, sys, CDLL, load_cdll, try_load_cdll, find_library, Path, warnings,
     _insert_pybind_names, _loaded_libs_cache, _detect_qt_version,
     _load_qt_library, _load_qt_libraries, _qt_version_cache, _setup_path,
     _setup_qt_plugin_path, _setup_linux_libraries)
