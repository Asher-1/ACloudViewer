Build from source on MacOS
=====================

## MACOS: [BUILD_SHELL](../../../scripts/build_macos.sh)

```
(VPN required)[fix librealsense downloading issues]
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
Remove build_realsense when call build_mac_wheel at scripts/build_macos_whl.sh

./scripts/build_macos.sh 2>&1 | tee build.log
```

## Debug wheel (MacOS)

```
1. lldb python

2. run -c "import cloudViewer"

3. bt
```

## Install dependencies

```
brew install gcc --without-multilib
```

## python env setup
```
cp .ci/conda_macos_cloudViewer.yml /tmp/conda_macos_cloudViewer.yml
sed -i "" "s/3.8/3.12/g" /tmp/conda_macos_cloudViewer.yml
conda env create -f /tmp/conda_macos_cloudViewer.yml
conda activate cloudViewer
```

## Building APP

```
PS: no opencv support due to some issues on local macos machine

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH

cd ACloudViewer
mkdir build_app
cd build_app
cmake   -DDEVELOPER_BUILD=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        -DBUILD_LIBREALSENSE=OFF \
        -DBUILD_AZURE_KINECT=OFF \
        -DBUILD_BENCHMARKS=ON \
        -DWITH_OPENMP=ON \
        -DWITH_IPP=OFF \
        -DWITH_SIMD=ON \
        -DUSE_SIMD=ON \
        -DPACKAGE=ON \
        -DUSE_PCL_BACKEND=ON \
        -DBUILD_WEBRTC=OFF \
        -DBUILD_OPENCV=OFF \
        -DUSE_SYSTEM_OPENCV=OFF \
        -DBUILD_RECONSTRUCTION=ON \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_COMMON_CUDA_ARCHS=ON \
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
        -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=OFF \
        -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=OFF \
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
        -DPLUGIN_STANDARD_QPOISSON_RECON=OFF \
        -DPOISSON_RECON_WITH_OPEN_MP=ON \
        -DPLUGIN_STANDARD_QRANSAC_SD=ON \
        -DPLUGIN_STANDARD_QSRA=ON \
        -DPLUGIN_STANDARD_3DMASC=OFF \
        -DPLUGIN_STANDARD_QTREEISO=OFF \
        -DPLUGIN_STANDARD_QVOXFALL=ON \
        -DPLUGIN_STANDARD_G3POINT=ON \
        -DPLUGIN_PYTHON=ON \
        -DBUILD_PYTHON_MODULE=ON \
        -DBUILD_WITH_CONDA=ON \
        -DCONDA_PREFIX=$CONDA_PREFIX \
        -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
        -DCMAKE_INSTALL_PREFIX=~/cloudViewer_install \
        -DCLOUDVIEWER_ML_ROOT=/Users/asher/develop/code/github/CloudViewer-ML \
        ..

Build: 
        make -j24
        make install -j24
```

## Building wheel

```
cp .ci/conda_macos.yml /tmp/conda_macos.yml
sed -i "" "s/3.8/3.12/g" /tmp/conda_macos.yml
conda env create -f /tmp/conda_macos.yml
conda activate python3.12

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh
install_python_dependencies with-unit-test purge-cache
```

```
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH

cd ACloudViewer
mkdir build
cd build

cmake -DDEVELOPER_BUILD=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBREALSENSE=ON \
      -DBUILD_AZURE_KINECT=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_OPENCV=OFF \
      -DWITH_OPENMP=ON \
      -DWITH_IPP=OFF \
      -DWITH_SIMD=ON \
      -DUSE_SIMD=ON \
      -DCVCORELIB_SHARED=ON \
      -DCVCORELIB_USE_CGAL=ON \
      -DCVCORELIB_USE_QT_CONCURRENT=ON \
      -DUSE_PCL_BACKEND=OFF \
      -DBUILD_FILAMENT_FROM_SOURCE=OFF \
      -DBUILD_WEBRTC=OFF \
      -DBUILD_JUPYTER_EXTENSION=OFF \
      -DBUILD_RECONSTRUCTION=ON \
      -DBUILD_CUDA_MODULE=OFF \
      -DBUILD_COMMON_CUDA_ARCHS=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_CLOUDVIEWER_ML=ON \
      -DBUILD_WITH_CONDA=ON \
      -DCONDA_PREFIX=$CONDA_PREFIX \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
      -DCMAKE_INSTALL_PREFIX=~/cloudViewer_install \
      -DCLOUDVIEWER_ML_ROOT=/Users/asher/develop/code/github/CloudViewer-ML \
      ..

make "-j$(nproc)" python-package
make "-j$(nproc)" pip-package
make "-j$(nproc)" install-pip-package
python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"
```

## Install

To install CloudViewer C++ library:

    	make install

To link a C++ project against the CloudViewer C++ library, please refer to
:ref:`create_cplusplus_project`.


To install CloudViewer Python library, build one of the following options:

    # Activate the virtualenv first
    # Install pip package in the current python environment
    make install-pip-package

    # Create Python package in build/lib
    make python-package

    # Create pip wheel in build/lib
    # This creates a .whl file that you can install manually.
    make pip-package

    # Create conda package in build/lib
    # This creates a .tar.bz2 file that you can install manually.
    make conda-package

Finally, verify the python installation with:

	python -c "import cloudViewer"

    :: Activate the virtualenv first
    :: Install pip package in the current python environment
    cmake --build . --config Release --target install-pip-package

    :: Create Python package in build/lib
    cmake --build . --config Release --target python-package

    :: Create pip package in build/lib
    :: This creates a .whl file that you can install manually.
    cmake --build . --config Release --target pip-package

    :: Create conda package in build/lib
    :: This creates a .tar.bz2 file that you can install manually.
    cmake --build . --config Release --target conda-package

Finally, verify the Python installation with:

    python -c "import cloudViewer; print(cloudViewer)"


## Compilation options
-------------------

OpenMP

We automatically detect if the C++ compiler supports OpenMP and compile CloudViewer
with it if the compilation option ``WITH_OPENMP`` is ``ON``.
OpenMP can greatly accelerate computation on a multi-core CPU.

The default LLVM compiler on OS X does not support OpenMP.
A workaround is to install a C++ compiler with OpenMP support, such as ``gcc``,
then use it to compile CloudViewer. For example, starting from a clean build
directory, run

    brew install gcc --without-multilib
    cmake -DCMAKE_C_COMPILER=gcc-6 -DCMAKE_CXX_COMPILER=g++-6 ..
    make -j

note:: This workaround has some compatibility issues with the source code of
    GLFW included in ``3rdparty``.
    Make sure CloudViewer is linked against GLFW installed on the OS.

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


## Unit test
---------

To build and run C++ unit tests:

    cmake -DBUILD_UNIT_TESTS=ON ..
    make -j
    ./bin/tests


To run Python unit tests:

    # Activate virtualenv first
    pip install pytest
    make install-pip-package
    pytest ../python/test

