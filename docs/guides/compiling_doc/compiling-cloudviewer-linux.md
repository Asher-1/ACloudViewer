# Building ACloudViewer from Source on Linux (Ubuntu)

> **Quick start with Docker?** See [docker/README.md](https://github.com/Asher-1/ACloudViewer/blob/main/docker/README.md) — no local setup required.
>
> ```bash
> ./docker/build-release.sh        # system-packages build
> ./docker/build-release-conda.sh  # conda-based build
> ```
>
> **One-click local builder (Ubuntu, pyenv path):** the script below automates
> the whole Option A flow for a **fresh machine** — apt packages (skips what is
> already installed), pyenv + Python 3.12, Vulkan SDK, VTK/PCL (prebuilt
> tarball via `docker/build_vtk_pcl_deps.sh consume`, source-build fallback),
> configure, build and the `.run` installer:
>
> ```bash
> bash docs/guides/compiling_doc/setup_and_build_acloudviewer_linux.sh --jobs 24
> ```
>
> **AI plugins (qDA3 / qDeepLSD / qFaceDetect / qFreeSplatter / qLightGlue):** see [docs/guides/plugins/](../../guides/plugins/README.md) for usage; enable with `-DAICore_ENABLED=ON -DAICore_USE_VULKAN=ON` (Linux default) plus `-DPLUGIN_STANDARD_QDA3=ON -DPLUGIN_STANDARD_QDEEPLSD=ON -DPLUGIN_STANDARD_QFACEDETECT=ON -DPLUGIN_STANDARD_QFREESPLATTER=ON -DPLUGIN_STANDARD_QLIGHTGLUE=ON`.

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
    - [Vulkan / AICore GPU (default)](#vulkan--aicore-gpu-default)
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
| **AICore GPU (default)** | **Vulkan** — `AICore_USE_VULKAN=ON` on Linux (CMake default); run `util/vulkan/install_vulkan_env.sh` or `install_deps_ubuntu.sh` for build-time SDK/glslc |
| **GPU (optional)** | CUDA toolkit ≥ 11.8 for optional `-DAICore_USE_CUDA=ON` / `-DBUILD_CUDA_MODULE=ON` |

---

## Option A — Build without Conda (System Packages Only)

> Use this path if you do **not** want or need Conda. All dependencies come from
> `apt` and `pyenv`. This is the recommended path for CI and clean environments.
>
> **Do NOT use Option A if you have a Conda environment activated.**
> Conda Python ships shared libraries (`.so`) while this path resolves `pyenv`
> static libraries (`.a`). Using `-DBUILD_WITH_CONDA=OFF` with a Conda Python
> will fail with *"Cannot find the library …/libpythonX.Y.a"*.
> If you are using Conda, skip to **[Option B](#option-b--build-with-conda)**.

### A1. Install system dependencies

```bash
# From the ACloudViewer repository root:
utils/install_deps_ubuntu.sh assume-yes
```

This script installs all required system packages (`xorg-dev`, `libglu1-mesa-dev`,
`ninja-build`, `libtbb-dev`, `libvulkan-dev`, etc.), adjusts clang/libc++ versions per
Ubuntu release, and runs `util/vulkan/install_vulkan_env.sh` to install the **Vulkan build
environment** (LunarG SDK headers, `glslc`, SPIR-V headers) used by AICore/qDA3.

> **Note:** `install_deps_ubuntu.sh` does **not** install the Qt5 development
> packages, but the APP build needs them. Install the same list the CI image
> uses (idempotent — already-installed packages are skipped):
>
> ```bash
> sudo apt-get install -y qtbase5-dev libqt5svg5-dev libqt5opengl5-dev \
>     qttools5-dev qttools5-dev-tools libqt5websockets5-dev \
>     libqt5xmlpatterns5-dev libqt5x11extras5-dev qtdeclarative5-dev \
>     qtdeclarative5-dev-tools libqt5quickcontrols2-5 libqt5networkauth5-dev \
>     qt5-image-formats-plugins qttranslations5-l10n libxxf86vm-dev libudev-dev
> ```

Reload the generated env in new shells before `cmake`:

```bash
source "${HOME}/.local/share/acloudviewer/acloudviewer-vulkan-env.sh"
```

If CMake later complains about missing packages, also run:

```bash
sudo apt install libxxf86vm-dev libudev-dev
```

### A2. Set up Python (pyenv)

```bash
export PYENV_ROOT=~/.pyenv
export PYTHON_VERSION=3.12
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"

# Install pyenv itself only when missing; `-s` skips versions that already
# exist (note: a partial version like 3.12 resolves to the *latest* 3.12.x,
# so an already-built 3.12.12 is reused through the symlink below).
command -v pyenv >/dev/null 2>&1 || curl https://pyenv.run | bash
pyenv update \
    && pyenv install -s $PYTHON_VERSION \
    && pyenv global $PYTHON_VERSION \
    && pyenv rehash
# Pin "3.12" to the newest installed 3.12.x build (-sfn keeps this re-runnable).
_real=$(ls -d "$PYENV_ROOT"/versions/${PYTHON_VERSION}.* 2>/dev/null | sort -V | tail -1 | xargs basename)
if [[ -n "${_real}" && "${_real}" != "${PYTHON_VERSION}" ]]; then
    ln -sfn "$PYENV_ROOT/versions/${_real}" "$PYENV_ROOT/versions/${PYTHON_VERSION}"
fi

python --version && pip --version
```

> **VTK / PCL:** the APP build with `USE_VTK_BACKEND=ON` and
> `BUILD_RECONSTRUCTION=ON` needs system VTK and PCL. On Ubuntu the apt
> versions are older than what CI uses; deploy the CI-pinned versions
> (VTK 9.3.1 + PCL 1.14.1, installed under `/usr/local`) with
> `docker/build_vtk_pcl_deps.sh consume` — it downloads a prebuilt tarball
> when available and falls back to a source build. The one-click script above
> handles this step (and skips it when VTK/PCL are already installed).

### A3. Build the APP (GUI + CLI)

> **Note:** Qt 6 is only supported on Ubuntu 24.04+. On 20.04/22.04 set `-DUSE_QT6=OFF`.

```bash
# All Option A commands below assume the repository root as the working
# directory (not the parent!):
cd ACloudViewer   # skip if you are already at the repository root
CLOUDVIEWER_SOURCE_ROOT=$(pwd)

# Mandatory when PLUGIN_PYTHON=ON (the configure step hard-fails without
# these modules): install the release deps into the pyenv Python.
python -m pip install -r \
    "${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt"

# Set your Qt installation path — cmake needs a prefix that actually contains
# the Qt5 cmake configs:
#   Ubuntu apt (qtbase5-dev): /usr            (configs at
#       /usr/lib/<arch>/cmake/Qt5 — NOT /usr/lib/x86_64-linux-gnu/qt5, which
#       only holds plugins/qml)
#   Qt online installer:      /opt/Qt/5.15.2/gcc_64  (or /opt/Qt5.14.2/5.14.2/gcc_64)
QT_DIR="/opt/Qt5.14.2/5.14.2/gcc_64"

mkdir -p build_app && cd build_app

# AICore (qDA3 / qFreeSplatter / qLightGlue): Vulkan is ON by default on Linux.
source "${HOME}/.local/share/acloudviewer/acloudviewer-vulkan-env.sh"
# nvcc must be reachable through PATH: BUILD_CUDA_MODULE=ON enables the CUDA
# language in the parent project and the ggml CUDA backend reuses it.
export PATH="/usr/local/cuda/bin:${PATH}"

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
    -DAICore_USE_VULKAN=ON \
    -DAICore_USE_CUDA=ON \
    -DAICore_BUNDLE_CUDA_RUNTIME=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QDEEPLSD=ON \
    -DPLUGIN_STANDARD_QFACEDETECT=ON \
    -DPLUGIN_STANDARD_QFREESPLATTER=ON \
    -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
    -DPLUGIN_STANDARD_QRFDETR=ON \
    -DPLUGIN_STANDARD_QRMBG=ON \
    -DPLUGIN_STANDARD_QYOLO=ON \
    -DPLUGIN_STANDARD_QSAM3=ON \
    -DPLUGIN_STANDARD_QTRELLIS=ON \
    -DPLUGIN_PYTHON=ON \
    -DBUILD_PYTHON_MODULE=ON \
    ..

make -j"$(nproc)"          # or -j24 on large machines; reduce if the linker is OOM-killed
make install -j"$(nproc)"
```

> **Why `BUILD_CUDA_MODULE=ON`:** `AICore_USE_CUDA=ON` requires the parent
> project to have the CUDA language enabled — the ggml ExternalProject
> inherits `CMAKE_CUDA_COMPILER` from it. With `BUILD_CUDA_MODULE=OFF` the
> parent never finds nvcc and `ext_ggml` configure fails with
> `No CMAKE_CUDA_COMPILER could be found` (an empty
> `-DCMAKE_CUDA_COMPILER=` disables the automatic search). Keep both ON
> together, or drop `AICore_USE_CUDA` and rely on Vulkan.

### A4. Build the Python wheel

```bash
cd ACloudViewer   # from the repository root

# CloudViewer-ML provides the ML ops Python wrappers bundled into the wheel:
[[ -d ~/develop/code/github/CloudViewer-ML ]] || \
    git clone --depth 1 https://github.com/Asher-1/CloudViewer-ML.git \
        ~/develop/code/github/CloudViewer-ML -b main
export CLOUDVIEWER_ML_ROOT=~/develop/code/github/CloudViewer-ML
# export CLOUDVIEWER_ML_ROOT=~/develop/code/github/CloudViewer/CloudViewer-ML

# Source CI utilities (run from the repository root)
CLOUDVIEWER_SOURCE_ROOT=$(pwd)
source "${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh"

export BUILD_PYTORCH_OPS=ON
install_python_dependencies with-cuda with-jupyter with-unit-test

# (Optional) Deploy Node.js + Yarn for Jupyter extension
curl -fsSL https://deb.nodesource.com/setup_25.x | sudo bash - \
    && sudo apt-get install -y nodejs \
    && sudo npm install -g yarn

mkdir -p build && cd build

source "${HOME}/.local/share/acloudviewer/acloudviewer-vulkan-env.sh"

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
    -DAICore_ENABLED=ON \
    -DAICore_USE_VULKAN=ON \
    -DAICore_USE_CUDA=ON \
    -DAICore_BUNDLE_CUDA_RUNTIME=ON \
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

> **Important:** The default `conda_cloudViewer.yml` ships with `python=3.10`.
> ACloudViewer requires **Python 3.10 – 3.13**. The `sed` command below
> replaces `3.10` with your chosen version. If you already have a `cloudViewer`
> Conda env with Python < 3.10, remove it first
> (`conda env remove -n cloudViewer`) and recreate it.

```bash
PYTHON_VERSION=3.12
cp .ci/conda_cloudViewer.yml /tmp/conda_cloudViewer.yml
sed -i "s/3.10/${PYTHON_VERSION}/g" /tmp/conda_cloudViewer.yml

conda env create -f /tmp/conda_cloudViewer.yml
conda activate cloudViewer

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
export PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH"
```

### B3. Build the APP (GUI + CLI)

> **Note:** Qt 6 is only supported on Ubuntu 24.04+. On 20.04/22.04 set `-DUSE_QT6=OFF`.
>
> **Note:** Conda builds **must** use `-DBUILD_WITH_CONDA=ON -DCONDA_PREFIX=$CONDA_PREFIX
> -DCMAKE_PREFIX_PATH=$CONDA_PREFIX`. Do **not** copy the Option A cmake block
> (which uses `-DBUILD_WITH_CONDA=OFF` and explicit `Python3_LIBRARY` paths).
> Using `$CONDA_PREFIX/lib` instead of `$CONDA_PREFIX` for `CMAKE_PREFIX_PATH`
> will cause CMake to miss the Conda Qt and pick up any system/standalone Qt
> installation (e.g. `/opt/Qt`), leading to ABI mismatches at build time.

```bash
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/ >/dev/null 2>&1 && pwd)"

# (Optional) Install Python plugin requirements
python -m pip install -r \
    "${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt"

cd ACloudViewer
mkdir -p build_app && cd build_app

source "${HOME}/.local/share/acloudviewer/acloudviewer-vulkan-env.sh"

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/install \
    -DBUILD_WITH_CONDA=ON \
    -DCONDA_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
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
    -DAICore_USE_VULKAN=ON \
    -DAICore_USE_CUDA=ON \
    -DAICore_BUNDLE_CUDA_RUNTIME=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QDEEPLSD=ON \
    -DPLUGIN_STANDARD_QFACEDETECT=ON \
    -DPLUGIN_STANDARD_QFREESPLATTER=ON \
    -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
    -DPLUGIN_STANDARD_QRFDETR=ON \
    -DPLUGIN_STANDARD_QRMBG=ON \
    -DPLUGIN_STANDARD_QYOLO=ON \
    -DPLUGIN_STANDARD_QSAM3=ON \
    -DPLUGIN_STANDARD_QTRELLIS=ON \
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
sed -i "s/3.10/${PYTHON_VERSION}/g" /tmp/conda_linux.yml

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

source "${HOME}/.local/share/acloudviewer/acloudviewer-vulkan-env.sh"

cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=~/install \
    -DBUILD_WITH_CONDA=ON \
    -DCONDA_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
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
    -DAICore_ENABLED=ON \
    -DAICore_USE_VULKAN=ON \
    -DAICore_USE_CUDA=ON \
    -DAICore_BUNDLE_CUDA_RUNTIME=ON \
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

> **Note:** `util/ci_utils.sh` also defaults `AICore_USE_VULKAN=ON` on Linux when you
> use `build_pip_package` / wheel helpers (`without_vulkan` disables it).

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

### Vulkan / AICore GPU (default)

On Linux, **AICore Auto device order is Vulkan → CPU**. CMake defaults
`-DAICore_USE_VULKAN=ON` when `AICore_ENABLED=ON`. Build-time tools (LunarG Vulkan SDK
headers, `glslc`, SPIR-V headers) are **not** shipped in the installer; only
`libggml-vulkan.so` is bundled. End users need a working Vulkan ICD/driver (or fall back
to CPU).

```bash
# One-shot setup (also run by install_deps_ubuntu.sh)
util/vulkan/install_vulkan_env.sh
source "${HOME}/.local/share/acloudviewer/acloudviewer-vulkan-env.sh"

cmake -DAICore_ENABLED=ON \
      -DAICore_USE_VULKAN=ON \
      -DAICore_USE_CUDA=ON \
      -DAICore_BUNDLE_CUDA_RUNTIME=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QDEEPLSD=ON \
      -DPLUGIN_STANDARD_QFACEDETECT=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
      -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
      -DPLUGIN_STANDARD_QRFDETR=ON \
      -DPLUGIN_STANDARD_QRMBG=ON \
      -DPLUGIN_STANDARD_QYOLO=ON \
      -DPLUGIN_STANDARD_QSAM3=ON \
      -DPLUGIN_STANDARD_QTRELLIS=ON \
      ..
```

CPU-only machine (no Vulkan SDK): `-DAICore_USE_VULKAN=OFF`.

Optional explicit CUDA for ggml (developer builds, not the portable Auto path):

```bash
cmake -DAICore_USE_CUDA=ON -DAICore_BUNDLE_CUDA_RUNTIME=ON ..
```

See [BUILD.md](https://github.com/Asher-1/ACloudViewer/blob/main/BUILD.md) for build-time vs runtime dependency tables.
`AICore_BUNDLE_CUDA_RUNTIME=ON` bundles `libcublas.so.*` under
`lib/cuda-runtime/` so the installer runs on driver-only machines: `cudart_static`
eliminates `libcudart.so.*`, but ggml hardwires non-quantized matmul to cuBLAS,
so `libggml-cuda.so` always needs `libcublas.so.*` at runtime. `ACloudViewer.sh`
auto-adds the bundled dir to `LD_LIBRARY_PATH`.

> **Portable deployment on Linux**: With `-DAICore_USE_CUDA=ON` +
> `-DAICore_BUNDLE_CUDA_RUNTIME=ON`, the only runtime requirement is the NVIDIA
> driver plus the bundled `lib/cuda-runtime/` (no CUDA Toolkit). See
> [CUDA / GPU](#cuda--gpu) below.

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
| `Cannot find the library "…/libpythonX.Y.a"` (Conda) | You used `-DBUILD_WITH_CONDA=OFF` with a Conda Python. Use [Option B](#option-b--build-with-conda) with `-DBUILD_WITH_CONDA=ON`. Also ensure your Conda env has Python >= 3.10 (see [B2](#b2-create-the-conda-environment)) |
| `lrelease: undefined symbol: _ZdlPvm` or wrong Qt tools used (Conda) | CMake found a standalone Qt (e.g. `/opt/Qt`) instead of the Conda Qt. Set `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX` (**not** `$CONDA_PREFIX/lib`). Delete `build_app/` and re-run cmake |
| Segfault on `import cloudViewer` | ABI mismatch — see [CXX ABI compatibility](#cxx-abi-compatibility) |
| CUDA not detected | Install CUDA toolkit and verify `nvcc -V` works |
| `AICore_USE_VULKAN=ON but Vulkan dependencies are missing` (ggml may print `GGML_USE_VULKAN` internally) | Run `util/vulkan/install_vulkan_env.sh`, then `source ~/.local/share/acloudviewer/acloudviewer-vulkan-env.sh` before `cmake` |
| `ext_ggml` configure fails with `No CMAKE_CUDA_COMPILER could be found` | You combined `AICore_USE_CUDA=ON` with `BUILD_CUDA_MODULE=OFF`. Enable both together (see [A3](#a3-build-the-app-gui--cli)) or drop `AICore_USE_CUDA` and rely on Vulkan |
| `BundleGgmlCudaRuntime failed: no CUDA runtime libraries were bundled` | The CUDA toolkit lib dir is not registered in `ldconfig`, so `ldd` reports `libcublas.so.12 => not found` and the bundler finds nothing. Fixed in `scripts/platforms/linux/bundle_cuda_runtime.sh` (unresolved sonames are now searched in `EXTRA_LIB_DIRS`); alternatively `export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH` before `make install` |
| `'begin' was not declared in this scope` in `Reconstruction/src/base/frame.cc` (or Eigen brace-init errors) | An old system-wide Eigen (< 3.4) shadows the bundled 3.4 headers used by the prebuilt VTK/PCL. Align it: `sudo rsync -a --delete build_app/external/include/eigen3/ /usr/local/include/eigen3/` (after the first configure), then rebuild |
| `libggml-vulkan.so` missing in wheel | Reconfigure with `-DAICore_USE_VULKAN=ON` and rebuild; do not use `without_vulkan` in `ci_utils.sh` |
| `clang: not found` on Ubuntu 20.04 | Run `install_deps_ubuntu.sh` — it installs version-specific clang |
