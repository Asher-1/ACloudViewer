# Building ACloudViewer from Source on macOS

> **Automated build script:** [`scripts/build_macos.sh`](../../../scripts/build_macos.sh)
>
> ```bash
> ./scripts/build_macos.sh 2>&1 | tee build.log
> ```

---

## Table of Contents

- [Building ACloudViewer from Source on macOS](#building-acloudviewer-from-source-on-macos)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Install System Dependencies](#install-system-dependencies)
  - [Set Up Python (Conda)](#set-up-python-conda)
  - [Building the APP (GUI + CLI)](#building-the-app-gui--cli)
  - [Building the Python Wheel](#building-the-python-wheel)
    - [1. Create the wheel Conda environment](#1-create-the-wheel-conda-environment)
    - [2. Install Python dependencies](#2-install-python-dependencies)
    - [3. Configure and build](#3-configure-and-build)
  - [Testing](#testing)
    - [Debug a wheel (LLDB)](#debug-a-wheel-lldb)
    - [Unit tests from CMake](#unit-tests-from-cmake)
    - [Python unit tests](#python-unit-tests)
  - [Installation](#installation)
    - [C++ library](#c-library)
    - [Python library](#python-library)
  - [Compilation Options Reference](#compilation-options-reference)
    - [OpenMP on macOS](#openmp-on-macos)
    - [ML Module (PyTorch)](#ml-module-pytorch)
    - [CUDA / GPU](#cuda--gpu)
  - [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Item             | Requirement                                       |
| ---------------- | ------------------------------------------------- |
| **OS**           | macOS 13+ (Intel or Apple Silicon)                |
| **Xcode CLI**    | `xcode-select --install`                          |
| **CMake**        | ≥ 3.20 (`brew install cmake`)                     |
| **Python**       | 3.10 – 3.12 (via Conda)                           |
| **Conda**        | Miniconda or Anaconda                              |
| **Homebrew**     | https://brew.sh                                    |

---

## Install System Dependencies

```bash
brew install gcc --without-multilib
```

> **VPN note:** If you are behind a firewall, librealsense downloads may fail.
> Set proxy environment variables before building:
>
> ```bash
> export https_proxy=http://127.0.0.1:7890
> export http_proxy=http://127.0.0.1:7890
> export all_proxy=socks5://127.0.0.1:7890
> ```

---

## Set Up Python (Conda)

```bash
PYTHON_VERSION=3.12

cp .ci/conda_macos_cloudViewer.yml /tmp/conda_macos_cloudViewer.yml
sed -i "" "s/3.8/${PYTHON_VERSION}/g" /tmp/conda_macos_cloudViewer.yml

conda env create -f /tmp/conda_macos_cloudViewer.yml
conda activate cloudViewer
```

Export Conda paths so CMake can find Qt, VTK, and other libraries:

```bash
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH"
```

---

## Building the APP (GUI + CLI)

> **qSIBR note:** The CI workflow sets `PLUGIN_STANDARD_QSIBR=OFF` on macOS.
> The example below matches that default. To experiment locally, add
> `-DPLUGIN_STANDARD_QSIBR=ON`.

```bash
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/ >/dev/null 2>&1 && pwd)"

# (Optional) Install Python plugin requirements
python -m pip install -r \
    "${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt"

cd ACloudViewer
mkdir -p build_app && cd build_app

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/ACloudViewer/install \
    -DBUILD_WITH_CONDA=ON \
    -DCONDA_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_BENCHMARKS=OFF \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_SIMD=ON \
    -DUSE_SIMD=ON \
    -DUSE_QT6=OFF \
    -DPACKAGE=ON \
    -DUSE_VTK_BACKEND=ON \
    -DBUILD_WEBRTC=OFF \
    -DBUILD_OPENCV=ON \
    -DUSE_SYSTEM_OPENCV=OFF \
    -DBUILD_RECONSTRUCTION=ON \
    -DBUILD_CUDA_MODULE=OFF \
    -DBUILD_JUPYTER_EXTENSION=OFF \
    -DBUILD_LIBREALSENSE=OFF \
    -DBUILD_AZURE_KINECT=OFF \
    -DBUILD_PYTORCH_OPS=OFF \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_CLOUDVIEWER_ML=OFF \
    -DCVCORELIB_USE_CGAL=ON \
    -DCVCORELIB_SHARED=ON \
    -DCVCORELIB_USE_QT_CONCURRENT=ON \
    -DOPTION_USE_GDAL=OFF \
    -DOPTION_USE_DXF_LIB=ON \
    -DOPTION_USE_RANSAC_LIB=ON \
    -DOPTION_USE_SHAPE_LIB=ON \
    -DPLUGIN_IO_QDRACO=ON \
    -DPLUGIN_IO_QLAS=ON \
    -DPLUGIN_IO_QADDITIONAL=ON \
    -DPLUGIN_IO_QCORE=ON \
    -DPLUGIN_IO_QCSV_MATRIX=ON \
    -DPLUGIN_IO_QE57=ON \
    -DPLUGIN_IO_QMESH=ON \
    -DPLUGIN_IO_QPDAL=OFF \
    -DPLUGIN_IO_QPHOTOSCAN=ON \
    -DPLUGIN_IO_QRDB=OFF \
    -DPLUGIN_IO_QFBX=OFF \
    -DPLUGIN_IO_QSTEP=OFF \
    -DPLUGIN_STANDARD_QCORK=ON \
    -DPLUGIN_STANDARD_QJSONRPC=ON \
    -DPLUGIN_STANDARD_QCLOUDLAYERS=ON \
    -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON \
    -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON \
    -DPLUGIN_STANDARD_QANIMATION=ON \
    -DQANIMATION_WITH_FFMPEG_SUPPORT=ON \
    -DPLUGIN_STANDARD_QCANUPO=ON \
    -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON \
    -DPLUGIN_STANDARD_QCOMPASS=ON \
    -DPLUGIN_STANDARD_QCSF=ON \
    -DPLUGIN_STANDARD_QFACETS=ON \
    -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON \
    -DPLUGIN_STANDARD_QM3C2=ON \
    -DPLUGIN_STANDARD_QMPLANE=ON \
    -DPLUGIN_STANDARD_QPCL=ON \
    -DPLUGIN_STANDARD_QPCV=ON \
    -DPLUGIN_STANDARD_QPOISSON_RECON=OFF \
    -DPLUGIN_STANDARD_QRANSAC_SD=ON \
    -DPLUGIN_STANDARD_QSRA=ON \
    -DPLUGIN_STANDARD_3DMASC=OFF \
    -DPLUGIN_STANDARD_QTREEISO=OFF \
    -DPLUGIN_STANDARD_QVOXFALL=ON \
    -DPLUGIN_STANDARD_G3POINT=ON \
    -DPLUGIN_STANDARD_QSIBR=OFF \
    -DPLUGIN_PYTHON=ON \
    -DBUILD_PYTHON_MODULE=ON \
    ..

make -j"$(sysctl -n hw.logicalcpu)"
make install -j"$(sysctl -n hw.logicalcpu)"
```

---

## Building the Python Wheel

### 1. Create the wheel Conda environment

```bash
PYTHON_VERSION=3.12

cp .ci/conda_macos.yml /tmp/conda_macos.yml
sed -i "" "s/3.8/${PYTHON_VERSION}/g" /tmp/conda_macos.yml

conda env create -f /tmp/conda_macos.yml
conda activate python${PYTHON_VERSION}
```

### 2. Install Python dependencies

```bash
export CLOUDVIEWER_ML_ROOT=/Users/asher/develop/code/github/CloudViewer-ML

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/ >/dev/null 2>&1 && pwd)"
export BUILD_PYTORCH_OPS=ON

source "${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh"
install_python_dependencies with-unit-test purge-cache
```

> **zsh users:** If `source` fails in zsh, wrap it:
>
> ```bash
> bash -l -c "source util/ci_utils.sh && install_python_dependencies with-unit-test purge-cache"
> ```

### 3. Configure and build

```bash
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH"
export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_CUDA_MODULE=OFF
export BUILD_PYTORCH_OPS=ON
export BUILD_TENSORFLOW_OPS=OFF

cd ACloudViewer
mkdir -p build && cd build

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/ACloudViewer/install \
    -DBUILD_WITH_CONDA=ON \
    -DCONDA_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
    -DBUILD_SHARED_LIBS=OFF \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_LIBREALSENSE=ON \
    -DBUILD_AZURE_KINECT=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_OPENCV=OFF \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=OFF \
    -DWITH_SIMD=ON \
    -DUSE_SIMD=ON \
    -DUSE_QT6=OFF \
    -DCVCORELIB_SHARED=ON \
    -DCVCORELIB_USE_CGAL=ON \
    -DCVCORELIB_USE_QT_CONCURRENT=ON \
    -DUSE_VTK_BACKEND=OFF \
    -DBUILD_FILAMENT_FROM_SOURCE=OFF \
    -DBUILD_WEBRTC=OFF \
    -DBUILD_JUPYTER_EXTENSION=OFF \
    -DBUILD_RECONSTRUCTION=ON \
    -DBUILD_CUDA_MODULE=OFF \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_CLOUDVIEWER_ML=ON \
    -DCLOUDVIEWER_ML_ROOT="${CLOUDVIEWER_ML_ROOT}" \
    ..

make -j"$(sysctl -n hw.logicalcpu)" python-package
make -j"$(sysctl -n hw.logicalcpu)" pip-package
make -j"$(sysctl -n hw.logicalcpu)" install-pip-package

python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"
```

---

## Testing

```bash
cd "${CLOUDVIEWER_SOURCE_ROOT}"
source util/ci_utils.sh

export BUILD_PYTORCH_OPS=ON
export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_TENSORFLOW_OPS=OFF

# Run all tests (C++ + Python)
run_all_tests

# Or run them separately:
run_cpp_unit_tests     # C++ unit tests only
run_python_tests       # Python unit tests only

# Test a built wheel
test_wheel build/lib/python_package/pip_package/cloudviewer*
```

### Debug a wheel (LLDB)

```bash
lldb python3
# (lldb) run -c "import cloudViewer"
# (lldb) bt
```

### Unit tests from CMake

```bash
cd build
cmake -DBUILD_UNIT_TESTS=ON ..
make -j"$(sysctl -n hw.logicalcpu)"
./bin/tests
```

### Python unit tests

```bash
pip install pytest
make install-pip-package
pytest ../python/test
```

---

## Installation

### C++ library

```bash
cd build
make install
```

To link against the installed C++ library, see the [C++ project guide](../../create_cplusplus_project.rst).

### Python library

```bash
# Install directly into current environment
make install-pip-package

# — or build artifacts for distribution —
make python-package    # → build/lib/
make pip-package       # → .whl in build/lib/
make conda-package     # → .tar.bz2 in build/lib/

# Verify
python -c "import cloudViewer; print(cloudViewer.__version__)"
```

---

## Compilation Options Reference

### OpenMP on macOS

The default Apple Clang does **not** support OpenMP. Workaround:

```bash
brew install gcc --without-multilib
cmake -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 -DWITH_OPENMP=ON ..
make -j"$(sysctl -n hw.logicalcpu)"
```

> **Note:** This workaround may have compatibility issues with the GLFW source
> bundled in `3rdparty/`. Make sure CloudViewer links against system GLFW if
> you encounter errors.

### ML Module (PyTorch)

```bash
cmake -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_CLOUDVIEWER_ML=ON \
      -DCLOUDVIEWER_ML_ROOT=https://github.com/intel-isl/CloudViewer-ML.git \
      ..
make -j"$(sysctl -n hw.logicalcpu)" install-pip-package
```

### CUDA / GPU

macOS does **not** support CUDA since macOS 10.14+. Set `-DBUILD_CUDA_MODULE=OFF`.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `librealsense` download fails | Set `https_proxy` / `http_proxy` (see [Install System Dependencies](#install-system-dependencies)), or disable with `-DBUILD_LIBREALSENSE=OFF` |
| OpenMP not found | Install GCC via Homebrew and set `CMAKE_C_COMPILER` / `CMAKE_CXX_COMPILER` |
| `source` fails in zsh | Use `bash -l -c "source util/ci_utils.sh && ..."` |
| Conda environment not activated | Verify with `echo $CONDA_PREFIX` before running cmake |
| `make -j$(nproc)` fails | macOS uses `sysctl -n hw.logicalcpu` instead of `nproc` |
| Qt not found | Ensure Conda env is activated and `$CONDA_PREFIX/lib` is in `CMAKE_PREFIX_PATH` |
