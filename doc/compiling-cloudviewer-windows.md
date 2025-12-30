
Build from source On Windows
=====================

## WINDOWS: [BUILD_SHELL](../scripts/build_win.py)

```
python .\scripts\build_win.py
```

## python env setup
```
$env:CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path
Copy-Item (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows_cloudViewer.yml") -Destination "$env:TEMP\conda_windows_cloudViewer.yml"
(Get-Content "$env:TEMP\conda_windows_cloudViewer.yml") -replace "3.8", $env:PYTHON_VERSION | Set-Content "$env:TEMP\conda_windows_cloudViewer.yml"

conda env create -f "$env:TEMP\conda_windows_cloudViewer.yml"
conda activate cloudViewer

$env:GENERATOR = "Visual Studio 17 2022"
$env:ARCHITECTURE = "x64"
$env:NPROC = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors
$env:CLOUDVIEWER_INSTALL_DIR = "C:/dev/cloudViewer_install"
```


## Building App

```
mkdir build_app
cd build_app
conda activate cloudViewer
../scripts/setup_conda_env.ps1

cmake -G $env:GENERATOR -A $env:ARCHITECTURE `
    -DDEVELOPER_BUILD="OFF" `
    -DBUILD_EXAMPLES=OFF `
    -DBUILD_SHARED_LIBS=OFF `
    -DSTATIC_WINDOWS_RUNTIME=OFF `
    -DBUILD_CUDA_MODULE=ON `
    -DWITH_OPENMP=ON `
    -DWITH_SIMD=ON `
    -DUSE_SIMD=ON `
    -DPACKAGE=ON `
    -DBUILD_BENCHMARKS=OFF `
    -DBUILD_OPENCV=ON `
    -DBUILD_RECONSTRUCTION=ON `
    -DUSE_PCL_BACKEND=ON `
    -DWITH_PCL_NURBS=ON `
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
    -DQANIMATION_WITH_FFMPEG_SUPPORT=ON `
    -DPLUGIN_STANDARD_QCANUPO=ON `
    -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON `
    -DPLUGIN_STANDARD_QCOMPASS=ON `
    -DPLUGIN_STANDARD_QCSF=ON `
    -DPLUGIN_STANDARD_QFACETS=ON `
    -DPLUGIN_STANDARD_G3POINT=ON `
    -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON `
    -DPLUGIN_STANDARD_QM3C2=ON `
    -DPLUGIN_STANDARD_QMPLANE=ON `
    -DPLUGIN_STANDARD_QPCL=ON `
    -DPLUGIN_STANDARD_QPOISSON_RECON=ON `
    -DPOISSON_RECON_WITH_OPEN_MP=ON `
    -DPLUGIN_STANDARD_QSRA=ON `
    -DPLUGIN_STANDARD_3DMASC=ON `
    -DPLUGIN_STANDARD_QTREEISO=ON `
    -DPLUGIN_STANDARD_QVOXFALL=ON `
    -DBUILD_WITH_CONDA=ON `
    -DCONDA_PREFIX=$env:CONDA_PREFIX `
    -DCMAKE_PREFIX_PATH=$env:CONDA_LIB_DIR `
    -DEIGEN_ROOT_DIR="$env:EIGEN_ROOT_DIR" `
    -DPLUGIN_PYTHON=ON `
    -DBUILD_PYTHON_MODULE=ON `
    -DBUILD_UNIT_TESTS=OFF `
    -DCMAKE_INSTALL_PREFIX=$env:CLOUDVIEWER_INSTALL_DIR `
    ..

cmake --build . --config Release --verbose --parallel 48
cmake --build . --config Release --target ACloudViewer --verbose --parallel 48

cmake --install . --config Release --verbose

```

## Building wheel

```
$env:CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path
Copy-Item (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows.yml") -Destination "$env:TEMP\conda_windows.yml"
(Get-Content "$env:TEMP\conda_windows.yml") -replace "3.8", "3.12" | Set-Content "$env:TEMP\conda_windows.yml"

conda env create -f "$env:TEMP\conda_windows.yml"
conda activate python3.12

. (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT "util\ci_utils.ps1")
Install-PythonDependencies -options "with-cuda","with-torch","with-jupyter"

# deploy yarn for jupyter building
node --version
npm --version
npm install -g yarn
yarn --version
```

```
mkdir build
cd build
../scripts/setup_conda_env.ps1

cmake -G $env:GENERATOR -A $env:ARCHITECTURE `
        -DBUILD_SHARED_LIBS=OFF `
        -DDEVELOPER_BUILD=OFF `
        -DCMAKE_BUILD_TYPE=Release `
        -DUSE_SYSTEM_EIGEN3=ON `
        -DBUILD_AZURE_KINECT=ON `
        -DBUILD_LIBREALSENSE=ON `
        -DBUILD_UNIT_TESTS=OFF `
        -DBUILD_BENCHMARKS=OFF `
        -DBUILD_CUDA_MODULE=OFF `
        -DUSE_SIMD=ON `
        -DWITH_SIMD=ON `
        -DWITH_OPENMP=ON `
        -DWITH_IPP=ON `
        -DCVCORELIB_SHARED=ON `
        -DCVCORELIB_USE_CGAL=ON `
        -DCVCORELIB_USE_QT_CONCURRENT=ON `
        -DUSE_PCL_BACKEND=OFF `
        -DBUILD_RECONSTRUCTION=ON `
        -DBUILD_PYTORCH_OPS=ON `
        -DBUILD_TENSORFLOW_OPS=OFF `
        -DBUNDLE_CLOUDVIEWER_ML=ON `
        -DBUILD_JUPYTER_EXTENSION=ON `
        -DBUILD_FILAMENT_FROM_SOURCE=OFF `
        -DBUILD_WITH_CONDA=ON `
        -DCONDA_PREFIX=$env:CONDA_PREFIX `
        -DCMAKE_PREFIX_PATH=$env:CONDA_LIB_DIR `
        -DCMAKE_INSTALL_PREFIX=$env:CLOUDVIEWER_INSTALL_DIR `
        ..

# build without cuda
cmake --build . --target python-package --config Release --parallel $env:NPROC
cmake --build . --target pip-package --config Release --parallel $env:NPROC

# build with cuda
cmake -DBUILD_CUDA_MODULE=ON ..
cmake --build . --target pip-package --config Release --parallel $env:NPROC
```


## Compilation options
-------------------
ML Module

Warning: Due to incompatibilities in the cxx11_abi on Linux between PyTorch and TensorFlow, 
official Python wheels on Linux only support PyTorch, not TensorFlow.
The ML module consists of primitives like operators and layers as well as high
level code for models and pipelines. To build the operators and layers, set
``BUILD_PYTORCH_OPS=ON`` and/or ``BUILD_TENSORFLOW_OPS=ON``.  Don't forget to also
enable ``BUILD_CUDA_MODULE=ON`` for GPU support. To include the models and
pipelines from CloudViewer-ML in the python package, set ``BUNDLE_CLOUDVIEWER_ML=ON`` and
``CLOUDVIEWER_ML_ROOT`` to the CloudViewer-ML repository. You can directly download
CloudViewer-ML from GitHub during the build with
``CLOUDVIEWER_ML_ROOT=https://github.com/intel-isl/CloudViewer-ML.git``.

The following example shows the command for building the ops with GPU support
for all supported ML frameworks and bundling the high level CloudViewer-ML code.

    # In the build directory
    cmake -DBUILD_CUDA_MODULE=ON \
          -DBUILD_PYTORCH_OPS=ON \
          -DBUILD_TENSORFLOW_OPS=OFF \
          -DBUNDLE_CLOUDVIEWER_ML=ON \
          -DCLOUDVIEWER_ML_ROOT=https://github.com/intel-isl/CloudViewer-ML.git \
          ..
    # Install the python wheel with pip
    make -j install-pip-package

note::
    Importing Python libraries compiled with different CXX ABI may cause segfaults
    in regex. https://stackoverflow.com/q/51382355/1255535. By default, PyTorch
    and TensorFlow Python releases use the older CXX ABI; while when they are
    compiled from source, newer ABI is enabled by default.
	When releasing CloudViewer as a Python package, we set
    ``-DGLIBCXX_USE_CXX11_ABI=OFF`` and compile all dependencies from source,
    in order to ensure compatibility with PyTorch and TensorFlow Python releases.
	If you build PyTorch or TensorFlow from source or if you run into ABI
    compatibility issues with them, please:

1. Check PyTorch and TensorFlow ABI with
	
		python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
		python -c "import tensorflow; print(tensorflow.__cxx11_abi_flag__)"

2. Configure CloudViewer to compile all dependencies from source
   with the corresponding ABI version obtained from step 1.

After installation of the Python package, you can check CloudViewer ABI version
with:

        python -c "import cloudViewer; print(cloudViewer.pybind._GLIBCXX_USE_CXX11_ABI)"

To build CloudViewer with CUDA support, configure with:

        cmake -DBUILD_CUDA_MODULE=ON -DCMAKE_INSTALL_PREFIX=<cloudViewer_install_directory> ..

Please note that CUDA support is work in progress and experimental. For building CloudViewer with CUDA support, ensure that CUDA is properly installed by running following commands:

code-block:: bash

        nvidia-smi      # Prints CUDA-enabled GPU information
        nvcc -V         # Prints compiler version

If you see an output similar to ``command not found``, you can install CUDA toolkit by following the `official documentation. <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_

