# Building ACloudViewer from Source on Windows

> **Automated build script:** [`scripts/build_win.py`](../../../scripts/build_win.py)
>
> ```powershell
> python .\scripts\build_win.py
> ```

---

## Table of Contents

- [Building ACloudViewer from Source on Windows](#building-acloudviewer-from-source-on-windows)
  - [Table of Contents](#table-of-contents)
  - [Prerequisites](#prerequisites)
  - [Set Up Python (Conda)](#set-up-python-conda)
  - [Building the APP (GUI + CLI)](#building-the-app-gui--cli)
  - [Building the Python Wheel](#building-the-python-wheel)
    - [1. Create the wheel Conda environment](#1-create-the-wheel-conda-environment)
    - [2. Install Python dependencies](#2-install-python-dependencies)
    - [3. Configure and build](#3-configure-and-build)
  - [Testing](#testing)
    - [Unit tests from CMake](#unit-tests-from-cmake)
    - [Python unit tests](#python-unit-tests)
  - [Compilation Options Reference](#compilation-options-reference)
    - [CUDA / GPU](#cuda--gpu)
    - [ML Module (PyTorch)](#ml-module-pytorch)
  - [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Item               | Requirement                                                         |
| ------------------ | ------------------------------------------------------------------- |
| **OS**             | Windows 10/11 (64-bit)                                              |
| **Visual Studio**  | 2022 (with "Desktop development with C++" workload)                 |
| **CMake**          | ≥ 3.20 (bundled with VS or standalone)                              |
| **Python**         | 3.10 – 3.12 (via Conda)                                             |
| **Conda**          | Miniconda or Anaconda                                               |
| **Git**            | https://git-scm.com/download/win                                    |
| **GPU (optional)** | CUDA toolkit ≥ 11.8 for GPU builds                                  |

---

## Set Up Python (Conda)

Open a **PowerShell** or **Developer Command Prompt** and run:

```powershell
$env:PYTHON_VERSION = "3.12"
$env:CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path

Copy-Item (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows_cloudViewer.yml") `
    -Destination "$env:TEMP\conda_windows_cloudViewer.yml"

(Get-Content "$env:TEMP\conda_windows_cloudViewer.yml") `
    -replace "3.8", $env:PYTHON_VERSION |
    Set-Content "$env:TEMP\conda_windows_cloudViewer.yml"

conda env create -f "$env:TEMP\conda_windows_cloudViewer.yml"
conda activate cloudViewer
python -m pip install -r \
    "${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt"

```

Set commonly used environment variables:

```powershell
$env:GENERATOR       = "Visual Studio 17 2022"
$env:ARCHITECTURE    = "x64"
$env:NPROC           = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors
$env:CLOUDVIEWER_INSTALL_DIR = "C:\dev\ACloudViewer_install"
```

---

## Building the APP (GUI + CLI)

```powershell
mkdir build_app
cd build_app

conda activate cloudViewer
..\scripts\setup_conda_env.ps1

cmake -G $env:GENERATOR -A $env:ARCHITECTURE `
    -DDEVELOPER_BUILD=OFF `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$env:CLOUDVIEWER_INSTALL_DIR" `
    -DBUILD_WITH_CONDA=ON `
    -DCONDA_PREFIX=$env:CONDA_PREFIX `
    -DCMAKE_PREFIX_PATH=$env:CONDA_LIB_DIR `
    -DEIGEN_ROOT_DIR="$env:EIGEN_ROOT_DIR" `
    -DBUILD_UNIT_TESTS=OFF `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_BENCHMARKS=OFF `
    -DBUILD_SHARED_LIBS=OFF `
    -DSTATIC_WINDOWS_RUNTIME=OFF `
    -DWITH_OPENMP=ON `
    -DWITH_SIMD=ON `
    -DUSE_SIMD=ON `
    -DPACKAGE=ON `
    -DBUILD_OPENCV=ON `
    -DBUILD_RECONSTRUCTION=ON `
    -DUSE_VTK_BACKEND=ON `
    -DBUILD_CUDA_MODULE=ON `
    -DCVCORELIB_USE_CGAL=ON `
    -DCVCORELIB_SHARED=ON `
    -DCVCORELIB_USE_QT_CONCURRENT=ON `
    -DOPTION_USE_GDAL=OFF `
    -DOPTION_USE_DXF_LIB=ON `
    -DPLUGIN_IO_QDRACO=ON `
    -DPLUGIN_IO_QLAS=ON `
    -DPLUGIN_IO_QADDITIONAL=ON `
    -DPLUGIN_IO_QCORE=ON `
    -DPLUGIN_IO_QCSV_MATRIX=ON `
    -DPLUGIN_IO_QE57=ON `
    -DPLUGIN_IO_QMESH=ON `
    -DPLUGIN_IO_QPDAL=OFF `
    -DPLUGIN_IO_QPHOTOSCAN=ON `
    -DPLUGIN_IO_QRDB=ON `
    -DPLUGIN_IO_QRDB_FETCH_DEPENDENCY=ON `
    -DPLUGIN_IO_QFBX=ON `
    -DPLUGIN_IO_QSTEP=OFF `
    -DPLUGIN_STANDARD_QCORK=ON `
    -DPLUGIN_STANDARD_QJSONRPC=ON `
    -DPLUGIN_STANDARD_QCLOUDLAYERS=ON `
    -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON `
    -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON `
    -DPLUGIN_STANDARD_QANIMATION=ON `
    -DPLUGIN_STANDARD_QBROOM=ON `
    -DQANIMATION_WITH_FFMPEG_SUPPORT=ON `
    -DPLUGIN_STANDARD_QCANUPO=ON `
    -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON `
    -DPLUGIN_STANDARD_QCOMPASS=ON `
    -DPLUGIN_STANDARD_QCSF=ON `
    -DPLUGIN_STANDARD_QFACETS=ON `
    -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON `
    -DPLUGIN_STANDARD_QM3C2=ON `
    -DPLUGIN_STANDARD_QMPLANE=ON `
    -DPLUGIN_STANDARD_QPCL=ON `
    -DPLUGIN_STANDARD_QPCV=ON `
    -DPLUGIN_STANDARD_QPOISSON_RECON=ON `
    -DPLUGIN_STANDARD_QSRA=ON `
    -DPLUGIN_STANDARD_3DMASC=ON `
    -DPLUGIN_STANDARD_QTREEISO=ON `
    -DPLUGIN_STANDARD_QVOXFALL=ON `
    -DPLUGIN_STANDARD_G3POINT=ON `
    -DPLUGIN_STANDARD_QSIBR=ON `
    -DAICore_ENABLED=ON `
    -DPLUGIN_STANDARD_QDA3=ON `
    -DPLUGIN_STANDARD_QFREESPLATTER=ON `
    -DPLUGIN_PYTHON=ON `
    -DBUILD_PYTHON_MODULE=ON `
    ..

cmake --build . --config Release --parallel $env:NPROC
cmake --build . --config Release --target ACloudViewer --parallel $env:NPROC
cmake --install . --config Release
```

---

## Building the Python Wheel

### 1. Create the wheel Conda environment

```powershell
$env:PYTHON_VERSION = "3.12"
$env:CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path

Copy-Item (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows.yml") `
    -Destination "$env:TEMP\conda_windows.yml"

(Get-Content "$env:TEMP\conda_windows.yml") `
    -replace "3.8", $env:PYTHON_VERSION |
    Set-Content "$env:TEMP\conda_windows.yml"

conda env create -f "$env:TEMP\conda_windows.yml"
conda activate python${env:PYTHON_VERSION}
```

### 2. Install Python dependencies

```powershell
$env:BUILD_PYTORCH_OPS = "ON"
$env:CLOUDVIEWER_ML_ROOT = "C:\path\to\CloudViewer-ML"
. (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")
Install-PythonDependencies -options "with-cuda","with-torch","with-jupyter"

# (Optional) Deploy Node.js + Yarn for Jupyter extension
node --version
npm --version
npm install -g yarn
yarn --version
```

### 3. Configure and build

```powershell
mkdir build
cd build
..\scripts\setup_conda_env.ps1

$env:DEVELOPER_BUILD      = "OFF"
$env:BUILD_SHARED_LIBS    = "OFF"
$env:BUILD_PYTORCH_OPS    = "ON"
$env:BUILD_TENSORFLOW_OPS = "OFF"

cmake -G $env:GENERATOR -A $env:ARCHITECTURE `
    -DDEVELOPER_BUILD=OFF `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_INSTALL_PREFIX="$env:CLOUDVIEWER_INSTALL_DIR" `
    -DBUILD_WITH_CONDA=ON `
    -DCONDA_PREFIX=$env:CONDA_PREFIX `
    -DCMAKE_PREFIX_PATH=$env:CONDA_LIB_DIR `
    -DBUILD_SHARED_LIBS=OFF `
    -DBUILD_UNIT_TESTS=OFF `
    -DBUILD_BENCHMARKS=OFF `
    -DUSE_SYSTEM_EIGEN3=ON `
    -DBUILD_AZURE_KINECT=ON `
    -DBUILD_LIBREALSENSE=ON `
    -DBUILD_CUDA_MODULE=OFF `
    -DUSE_SIMD=ON `
    -DWITH_SIMD=ON `
    -DWITH_OPENMP=ON `
    -DWITH_IPP=ON `
    -DCVCORELIB_SHARED=ON `
    -DCVCORELIB_USE_CGAL=ON `
    -DCVCORELIB_USE_QT_CONCURRENT=ON `
    -DUSE_VTK_BACKEND=OFF `
    -DBUILD_RECONSTRUCTION=ON `
    -DBUILD_PYTORCH_OPS=ON `
    -DBUILD_TENSORFLOW_OPS=OFF `
    -DBUNDLE_CLOUDVIEWER_ML=ON `
    -DCLOUDVIEWER_ML_ROOT="${CLOUDVIEWER_ML_ROOT}" `
    -DBUILD_JUPYTER_EXTENSION=ON `
    -DBUILD_FILAMENT_FROM_SOURCE=OFF `
    ..

# Build without CUDA first
cmake --build . --target python-package --config Release --parallel $env:NPROC
cmake --build . --target pip-package    --config Release --parallel $env:NPROC

# (Optional) Enable CUDA and rebuild
cmake -DBUILD_CUDA_MODULE=ON ..
cmake --build . --target pip-package --config Release --parallel $env:NPROC
```

---

## Testing

```powershell
$env:CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path
$env:BUILD_PYTORCH_OPS    = "ON"
$env:DEVELOPER_BUILD      = "OFF"
$env:BUILD_SHARED_LIBS    = "OFF"
$env:BUILD_TENSORFLOW_OPS = "OFF"

. util\ci_utils.ps1

# Test wheel package
$wheelPath = Get-ChildItem build\lib\python_package\pip_package\cloudviewer*.whl |
    Select-Object -First 1
if ($wheelPath) {
    Test-Wheel -wheel_path $wheelPath.FullName
}

# Run all tests (C++ + Python)
Run-AllTests -wheel_path $wheelPath.FullName

# Or run them separately:
Run-CppUnitTests                                 # C++ only
Run-PythonTests -wheel_path $wheelPath.FullName   # Python only
```

### Unit tests from CMake

```powershell
cd build
cmake -DBUILD_UNIT_TESTS=ON ..
cmake --build . --config Release --parallel $env:NPROC
.\bin\Release\tests.exe
```

### Python unit tests

```powershell
pip install pytest
cmake --build . --config Release --target install-pip-package
pytest ..\python\test
```

---

## Compilation Options Reference

### CUDA / GPU

```powershell
cmake -DBUILD_CUDA_MODULE=ON `
      -DBUILD_COMMON_CUDA_ARCHS=ON `
      -DCMAKE_INSTALL_PREFIX="$env:CLOUDVIEWER_INSTALL_DIR" `
      ..
```

Verify CUDA is available:

```powershell
nvidia-smi   # GPU info
nvcc -V       # Compiler version
```

If these commands fail, install the CUDA toolkit via the
[official guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### ML Module (PyTorch)

> **Warning:** On Windows, official wheels only support PyTorch due to CXX ABI
> incompatibilities between PyTorch and TensorFlow.

```powershell
cmake -DBUILD_CUDA_MODULE=ON `
      -DBUILD_PYTORCH_OPS=ON `
      -DBUILD_TENSORFLOW_OPS=OFF `
      -DBUNDLE_CLOUDVIEWER_ML=ON `
      -DCLOUDVIEWER_ML_ROOT="https://github.com/intel-isl/CloudViewer-ML.git" `
      ..

cmake --build . --target install-pip-package --config Release --parallel $env:NPROC
```

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `MSBUILD : error MSB1009: Project file does not exist` | Ensure you run `cmake ..` from inside the `build` directory |
| Conda env not found after `conda activate` | Restart PowerShell or run `conda init powershell` first |
| `setup_conda_env.ps1` script fails | Verify `$env:CONDA_PREFIX` is set: `echo $env:CONDA_PREFIX` |
| Qt not found by CMake | Run `.\scripts\setup_conda_env.ps1` to set `CONDA_LIB_DIR` and `EIGEN_ROOT_DIR` |
| CUDA not detected | Install CUDA toolkit and verify `nvcc -V` in PowerShell |
| `parallel` flag ignored | Use `/m:N` for MSBuild or `--parallel N` for `cmake --build` |
| Long path errors | Enable Win32 long paths: `New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1` |
