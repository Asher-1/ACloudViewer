# Building ACloudViewer from Source on Linux (Ubuntu)

> **Quick start with Docker?** See [docker/README.md](../../../docker/README.md) — no local setup required.
>
> ```bash
> ./docker/build-release.sh        # system-packages build
> ./docker/build-release-conda.sh  # conda-based build
> ```

---

## Table of Contents

- [Building ACloudViewer from Source on Linux (Ubuntu)](#building-acloudviewer-from-source-on-linux-ubuntu)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Option A — Build without Conda (System Packages Only)](#option-a--build-without-conda-system-packages-only)
    - [A1. Install system dependencies](#a1-install-system-dependencies)
    - [A2. Set up Python (pyenv)](#a2-set-up-python-pyenv)
    - [A3. Build the APP (GUI + CLI)](#a3-build-the-app-gui--cli)
    - [A4. Build the Python wheel](#a4-build-the-python-wheel)
  - [Option B — Build with Conda](#option-b--build-with-conda)
    - [B1. Install system dependencies](#b1-install-system-dependencies)
    - [B2. Create the Conda environment](#b2-create-the-conda-environment)
    - [B3. Build the APP (GUI + CLI)](#b3-build-the-app-gui--cli)
    - [B4. Build the Python wheel](#b4-build-the-python-wheel)
  - [Testing](#testing)
    - [Debug a wheel (GDB)](#debug-a-wheel-gdb)
    - [Unit tests from CMake](#unit-tests-from-cmake)
    - [Python unit tests](#python-unit-tests)
  - [Installation](#installation)
    - [C++ library](#c-library)
    - [Python library](#python-library)
  - [Compilation Options Reference](#compilation-options-reference)
    - [CUDA / GPU](#cuda--gpu)
    - [ML Module (PyTorch / TensorFlow)](#ml-module-pytorch--tensorflow)
    - [CXX ABI compatibility](#cxx-abi-compatibility)
  - [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Item             | Requirement                                    |
| ---------------- | ---------------------------------------------- |
| **OS**           | Ubuntu 20.04 / 22.04 / 24.04                   |
| **CMake**        | ≥ 3.20                                         |
| **Python**       | 3.10 – 3.13                                    |
| **Compiler**     | GCC ≥ 9 or Clang (provided by `install_deps`)  |
| **GPU (optional)** | CUDA toolkit ≥ 11.8 for GPU builds           |

---

## Option A — Build without Conda (System Packages Only)

> Use this path if you do **not** want or need Conda. All dependencies come from
> `apt` and `pyenv`. This is the recommended path for CI and clean environments.

### A1. Install system dependencies

```bash
# From the ACloudViewer repository root:
utils/install_deps_ubuntu.sh assume-yes
```

This script installs all required system packages (`xorg-dev`, `libglu1-mesa-dev`,
`ninja-build`, `libtbb-dev`, etc.) and adjusts clang/libc++ versions per Ubuntu release.

If CMake later complains about missing packages, also run:

```bash
sudo apt install libxxf86vm-dev libudev-dev
```

### A2. Set up Python (pyenv)

```bash
export PYENV_ROOT=~/.pyenv
export PYTHON_VERSION=3.12
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"

curl https://pyenv.run | bash \
    && pyenv update \
    && pyenv install $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash \
    && ln -s "$PYENV_ROOT/versions/${PYTHON_VERSION}"* "$PYENV_ROOT/versions/${PYTHON_VERSION}"

python --version && pip --version
```

### A3. Build the APP (GUI + CLI)

> **Note:** Qt 6 is only supported on Ubuntu 24.04+. On 20.04/22.04 set `-DUSE_QT6=OFF`.

```bash
# Resolve Python paths for CMake
PYTHON_EXE=$(pyenv which python)
PYTHON_ROOT=$(python -c "import sysconfig, os; print(os.path.dirname(os.path.dirname(sysconfig.get_path('include'))))")
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_DIR=$(python -c "import sysconfig, os; libdir = sysconfig.get_config_var('LIBDIR'); print(os.path.realpath(libdir) if os.path.islink(libdir) else libdir)")
PYTHON_LIB_NAME=$(python -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))")
PYTHON_LIB="${PYTHON_LIB_DIR}/${PYTHON_LIB_NAME}"

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/ >/dev/null 2>&1 && pwd)"

# (Optional) Install Python plugin requirements
python -m pip install -r \
    "${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt"

# Set your Qt installation path — common locations:
#   Ubuntu apt:          /usr/lib/x86_64-linux-gnu/qt5
#   Qt online installer: /opt/qt515/lib/cmake  or  /opt/Qt/5.15.2/gcc_64
#   Custom build:        /path/to/your/qt5
QT_DIR="/usr/lib/x86_64-linux-gnu/qt5"

cd ACloudViewer
mkdir -p build_app && cd build_app

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/install \
    -DBUILD_WITH_CONDA=OFF \
    -DCMAKE_PREFIX_PATH="${QT_DIR}" \
    -DPython3_EXECUTABLE="${PYTHON_EXE}" \
    -DPython3_ROOT_DIR="${PYTHON_ROOT}" \
    -DPython3_LIBRARY="${PYTHON_LIB}" \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_SIMD=ON \
    -DUSE_SIMD=ON \
    -DUSE_QT6=OFF \
    -DUSE_VTK_BACKEND=ON \
    -DBUILD_WEBRTC=OFF \
    -DBUILD_OPENCV=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DBUILD_CUDA_MODULE=OFF \
    -DBUILD_JUPYTER_EXTENSION=OFF \
    -DBUILD_LIBREALSENSE=OFF \
    -DBUILD_AZURE_KINECT=OFF \
    -DBUILD_PYTORCH_OPS=OFF \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_CLOUDVIEWER_ML=OFF \
    -DPACKAGE=ON \
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
    -DPLUGIN_STANDARD_QBROOM=ON \
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
    -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
    -DPLUGIN_STANDARD_QRANSAC_SD=ON \
    -DPLUGIN_STANDARD_QSRA=ON \
    -DPLUGIN_STANDARD_3DMASC=ON \
    -DPLUGIN_STANDARD_QTREEISO=ON \
    -DPLUGIN_STANDARD_QVOXFALL=ON \
    -DPLUGIN_STANDARD_G3POINT=ON \
    -DPLUGIN_STANDARD_QSIBR=ON \
    -DAICore_ENABLED=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QFREESPLATTER=ON \
    -DPLUGIN_PYTHON=ON \
    -DBUILD_PYTHON_MODULE=ON \
    ..

make -j"$(nproc)"
make install -j"$(nproc)"
```

### A4. Build the Python wheel

```bash
cd ACloudViewer

# set CLOUDVIEWER_ML_ROOT path
export CLOUDVIEWER_ML_ROOT=~/develop/code/github/CloudViewer-ML
# export CLOUDVIEWER_ML_ROOT=~/develop/code/github/CloudViewer/CloudViewer-ML

# Source CI utilities
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/ >/dev/null 2>&1 && pwd)"
source "${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh"

export BUILD_PYTORCH_OPS=ON
install_python_dependencies with-cuda with-jupyter with-unit-test

# (Optional) Deploy Node.js + Yarn for Jupyter extension
curl -fsSL https://deb.nodesource.com/setup_25.x | sudo bash - \
    && sudo apt-get install -y nodejs \
    && sudo npm install -g yarn

mkdir -p build && cd build

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/install \
    -DBUILD_WITH_CONDA=OFF \
    -DCMAKE_PREFIX_PATH="${QT_DIR}" \
    -DPython3_EXECUTABLE="${PYTHON_EXE}" \
    -DPython3_ROOT_DIR="${PYTHON_ROOT}" \
    -DPython3_LIBRARY="${PYTHON_LIB}" \
    -DBUILD_LIBREALSENSE=ON \
    -DBUILD_AZURE_KINECT=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_SIMD=ON \
    -DUSE_SIMD=ON \
    -DUSE_QT6=OFF \
    -DCVCORELIB_SHARED=ON \
    -DCVCORELIB_USE_CGAL=ON \
    -DCVCORELIB_USE_QT_CONCURRENT=ON \
    -DUSE_VTK_BACKEND=OFF \
    -DBUILD_FILAMENT_FROM_SOURCE=OFF \
    -DBUILD_WEBRTC=ON \
    -DBUILD_JUPYTER_EXTENSION=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DBUILD_OPENCV=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_COMMON_CUDA_ARCHS=ON \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_CLOUDVIEWER_ML=ON \
    -DCLOUDVIEWER_ML_ROOT="${CLOUDVIEWER_ML_ROOT}" \
    ..

make -j"$(nproc)" python-package

# Enable CUDA then build the pip package
cmake -DBUILD_CUDA_MODULE=ON ..
make -j"$(nproc)" pip-package
make -j"$(nproc)" install-pip-package

python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"
```

---

## Option B — Build with Conda

> Use this path when you need libraries (Qt, VTK, CGAL, etc.) managed by Conda
> instead of system packages. Recommended for reproducible builds and
> environments where system packages may be outdated.

### B1. Install system dependencies

The base system tools are still needed even with Conda:

```bash
utils/install_deps_ubuntu.sh assume-yes
```

### B2. Create the Conda environment

```bash
PYTHON_VERSION=3.12
cp .ci/conda_cloudViewer.yml /tmp/conda_cloudViewer.yml
sed -i "s/3.8/${PYTHON_VERSION}/g" /tmp/conda_cloudViewer.yml

conda env create -f /tmp/conda_cloudViewer.yml
conda activate cloudViewer

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
export PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH"
```

### B3. Build the APP (GUI + CLI)

> **Note:** Qt 6 is only supported on Ubuntu 24.04+. On 20.04/22.04 set `-DUSE_QT6=OFF`.
```

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
    -DCMAKE_INSTALL_PREFIX=~/install \
    -DBUILD_WITH_CONDA=ON \
    -DCONDA_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_BENCHMARKS=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_SIMD=ON \
    -DUSE_SIMD=ON \
    -DUSE_QT6=OFF \
    -DUSE_VTK_BACKEND=ON \
    -DBUILD_WEBRTC=OFF \
    -DBUILD_OPENCV=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DBUILD_CUDA_MODULE=ON \
    -DBUILD_COMMON_CUDA_ARCHS=ON \
    -DBUILD_JUPYTER_EXTENSION=OFF \
    -DBUILD_LIBREALSENSE=OFF \
    -DBUILD_AZURE_KINECT=OFF \
    -DBUILD_PYTORCH_OPS=OFF \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_CLOUDVIEWER_ML=OFF \
    -DPACKAGE=ON \
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
    -DPLUGIN_STANDARD_QBROOM=ON \
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
    -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
    -DPLUGIN_STANDARD_QRANSAC_SD=ON \
    -DPLUGIN_STANDARD_QSRA=ON \
    -DPLUGIN_STANDARD_3DMASC=ON \
    -DPLUGIN_STANDARD_QTREEISO=ON \
    -DPLUGIN_STANDARD_QVOXFALL=ON \
    -DPLUGIN_STANDARD_G3POINT=ON \
    -DPLUGIN_STANDARD_QSIBR=ON \
    -DAICore_ENABLED=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QFREESPLATTER=ON \
    -DPLUGIN_PYTHON=ON \
    -DBUILD_PYTHON_MODULE=ON \
    ..

make -j"$(nproc)"
make install -j"$(nproc)"
```

### B4. Build the Python wheel

Export paths so CMake can discover Conda packages:

```bash
PYTHON_VERSION=3.12
cp .ci/conda_linux.yml /tmp/conda_linux.yml
sed -i "s/3.8/${PYTHON_VERSION}/g" /tmp/conda_linux.yml

conda env create -f /tmp/conda_linux.yml
conda activate python${PYTHON_VERSION}
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
export PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH"
```

```bash
# Ensure Conda env paths are exported (see B2)
export BUILD_PYTORCH_OPS=ON
export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_TENSORFLOW_OPS=OFF

# set CLOUDVIEWER_ML_ROOT path
export CLOUDVIEWER_ML_ROOT=~/develop/code/github/CloudViewer-ML
# export CLOUDVIEWER_ML_ROOT=~/develop/code/github/CloudViewer/CloudViewer-ML

cd ACloudViewer
mkdir -p build && cd build

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/install \
    -DBUILD_WITH_CONDA=ON \
    -DCONDA_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
    -DBUILD_LIBREALSENSE=ON \
    -DBUILD_AZURE_KINECT=ON \
    -DWITH_OPENMP=ON \
    -DWITH_IPP=ON \
    -DWITH_SIMD=ON \
    -DUSE_SIMD=ON \
    -DUSE_QT6=OFF \
    -DCVCORELIB_SHARED=ON \
    -DCVCORELIB_USE_CGAL=ON \
    -DCVCORELIB_USE_QT_CONCURRENT=ON \
    -DUSE_VTK_BACKEND=OFF \
    -DBUILD_FILAMENT_FROM_SOURCE=OFF \
    -DBUILD_WEBRTC=ON \
    -DBUILD_JUPYTER_EXTENSION=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DBUILD_OPENCV=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_COMMON_CUDA_ARCHS=ON \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_TENSORFLOW_OPS=OFF \
    -DBUNDLE_CLOUDVIEWER_ML=ON \
    -DCLOUDVIEWER_ML_ROOT="${CLOUDVIEWER_ML_ROOT}" \
    ..

make -j"$(nproc)" python-package

# Enable CUDA then build the pip package
cmake -DBUILD_CUDA_MODULE=ON ..
make -j"$(nproc)" pip-package
make -j"$(nproc)" install-pip-package

python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"
```

---

## Testing

```bash
cd "${CLOUDVIEWER_SOURCE_ROOT}"
source util/ci_utils.sh

# Run all tests (C++ + Python)
run_all_tests

# Or run them separately:
run_cpp_unit_tests     # C++ unit tests only
run_python_tests       # Python unit tests only

# Test a built wheel
test_wheel build/lib/python_package/pip_package/cloudviewer*
```

### Debug a wheel (GDB)

```bash
# Quick backtrace on import failure
gdb --batch --ex run --ex bt --ex quit --args python3 -c "import cloudViewer"

# Interactive session
gdb python3
# (gdb) run -c "import cloudViewer"
# (gdb) bt
```

### Unit tests from CMake

```bash
cd build
cmake -DBUILD_UNIT_TESTS=ON ..
make -j"$(nproc)"
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

### CUDA / GPU

```bash
cmake -DBUILD_CUDA_MODULE=ON \
      -DBUILD_COMMON_CUDA_ARCHS=ON \
      -DCMAKE_INSTALL_PREFIX=~/install \
      ..
```

Verify CUDA is available:

```bash
nvidia-smi   # GPU info
nvcc -V       # Compiler version
```

If these commands fail, install the CUDA toolkit via the
[official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

### ML Module (PyTorch / TensorFlow)

> **Warning:** On Linux, official Python wheels only support PyTorch due to
> CXX11 ABI incompatibilities between PyTorch and TensorFlow.

```bash
cmake -DBUILD_CUDA_MODULE=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_CLOUDVIEWER_ML=ON \
      -DCLOUDVIEWER_ML_ROOT=https://github.com/intel-isl/CloudViewer-ML.git \
      ..
make -j"$(nproc)" install-pip-package
```

### CXX ABI compatibility

If you build PyTorch or TensorFlow from source and encounter ABI issues:

```bash
# Check ABI of installed frameworks
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
python -c "import tensorflow; print(tensorflow.__cxx11_abi_flag__)"

# Check ABI of installed CloudViewer
python -c "import cloudViewer; print(cloudViewer.pybind._GLIBCXX_USE_CXX11_ABI)"
```

Set `-DGLIBCXX_USE_CXX11_ABI=OFF` (or `ON`) to match the frameworks you depend on.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `find_package` cannot find Qt5 | `sudo apt install qtbase5-dev`, then set `QT_DIR` to your Qt path (e.g. `/usr/lib/x86_64-linux-gnu/qt5`, `/opt/qt515`, or `/opt/Qt/5.15.2/gcc_64`) |
| Missing `libXxf86vm` or `libudev` | `sudo apt install libxxf86vm-dev libudev-dev` |
| `Python3_LIBRARY` not found | Provide explicit `-DPython3_EXECUTABLE` / `-DPython3_LIBRARY` (see [Option A](#a3-build-the-app-gui--cli)) |
| Segfault on `import cloudViewer` | ABI mismatch — see [CXX ABI compatibility](#cxx-abi-compatibility) |
| CUDA not detected | Install CUDA toolkit and verify `nvcc -V` works |
| `clang: not found` on Ubuntu 20.04 | Run `install_deps_ubuntu.sh` — it installs version-specific clang |
