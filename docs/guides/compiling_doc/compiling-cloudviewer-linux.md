Build from source on Ubuntu
=====================

[**Fast Docker build on Linux**](../../../docker/README.md)

LINUX:(Docker) [BUILD_SHELL](../../../docker/build-release.sh) and [BUILD_SHELL_CONDA](../../../docker/build-release-conda.sh)

```
./docker/build-release.sh

or

./docker/build-release-conda.sh
```

## Debug wheel (Linux)

```
1. gdb --batch --ex run --ex bt --ex quit --args python3 -c "import cloudViewer"

或者

2. gdb python3
3. run -c "import cloudViewer"

4. bt
```

## Install dependencies
```
utils/install_deps_ubuntu.sh assume-yes
```

## python env setup
```
export PYENV_ROOT=~/.pyenv PYTHON_VERSION=3.12
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"

curl https://pyenv.run | bash \
        && pyenv update \
        && pyenv install $PYTHON_VERSION \
        && pyenv global $PYTHON_VERSION \
        && pyenv rehash \
        && ln -s $PYENV_ROOT/versions/${PYTHON_VERSION}* $PYENV_ROOT/versions/${PYTHON_VERSION};
python --version && pip --version
```

## Building APP

```
Note: only Ubuntu24.04+ support QT6

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/ >/dev/null 2>&1 && pwd)"
# you can use PackageManager to install 3DFin==0.4.1 as python plugin (with qt5 support not latest version)
python -m pip install -r ${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt

cd ACloudViewer
mkdir build_app
cd build_app
cmake   -DDEVELOPER_BUILD=OFF \
        -DBUILD_UNIT_TESTS=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        -DBUILD_LIBREALSENSE=OFF \
        -DBUILD_AZURE_KINECT=OFF \
        -DBUILD_BENCHMARKS=ON \
        -DWITH_OPENMP=ON \
        -DWITH_IPP=ON \
        -DWITH_SIMD=ON \
        -DUSE_SIMD=ON \
        -DPACKAGE=ON \
        -DUSE_QT6=OFF \
        -DUSE_PCL_BACKEND=ON \
        -DBUILD_WEBRTC=OFF \
        -DBUILD_OPENCV=ON \
        -DBUILD_RECONSTRUCTION=ON \
        -DBUILD_CUDA_MODULE=ON \
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
        -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
        -DPOISSON_RECON_WITH_OPEN_MP=ON \
        -DPLUGIN_STANDARD_QRANSAC_SD=ON \
        -DPLUGIN_STANDARD_QSRA=ON \
        -DPLUGIN_STANDARD_3DMASC=ON \
        -DPLUGIN_STANDARD_QTREEISO=ON \
        -DPLUGIN_STANDARD_QVOXFALL=ON \
        -DPLUGIN_STANDARD_G3POINT=ON \
        -DPLUGIN_PYTHON=ON \
        -DBUILD_PYTHON_MODULE=ON \
        -DBUILD_WITH_CONDA=ON \
        -DCONDA_PREFIX=$CONDA_PREFIX \
        -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
        -DCMAKE_INSTALL_PREFIX=/home/asher/develop/code/github/CloudViewer/install \
        ..

Build: 
        make "-j$(nproc)"
        make install "-j$(nproc)"
```


## Building wheel

```
export PYENV_ROOT=~/.pyenv PYTHON_VERSION=3.12
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"

curl https://pyenv.run | bash \
        && pyenv update \
        && pyenv install $PYTHON_VERSION \
        && pyenv global $PYTHON_VERSION \
        && pyenv rehash \
        && ln -s $PYENV_ROOT/versions/${PYTHON_VERSION}* $PYENV_ROOT/versions/${PYTHON_VERSION};
python --version && pip --version

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/ >/dev/null 2>&1 && pwd)"

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"
export BUILD_PYTORCH_OPS=ON
install_python_dependencies with-cuda with-jupyter with-unit-test

```

### deploy yarn for jupyter building
```
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo bash - \
 && sudo apt-get install -y nodejs \
 && node --version \
 && sudo npm install -g yarn \
 && yarn --version
```

### fix cmake failed to find python and qt
```
PYTHON_EXE=$(pyenv which python)
PYTHON_ROOT=$(python -c "import sysconfig, os; print(os.path.dirname(os.path.dirname(sysconfig.get_path('include'))))")
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")
PYTHON_LIB_DIR=$(python -c "import sysconfig, os; libdir = sysconfig.get_config_var('LIBDIR'); print(os.path.realpath(libdir) if os.path.islink(libdir) else libdir)")
PYTHON_LIB_NAME=$(python -c "import sysconfig; print(sysconfig.get_config_var('LDLIBRARY'))")
PYTHON_LIB="${PYTHON_LIB_DIR}/${PYTHON_LIB_NAME}"
-DPython3_EXECUTABLE="${PYTHON_EXE}" \
-DPython3_ROOT_DIR="${PYTHON_ROOT}" \
-DPython3_LIBRARY="${PYTHON_LIB}" \
-DBUILD_WITH_CONDA=OFF \
-DCMAKE_PREFIX_PATH=/usr/lib/x86_64-linux-gnu/qt5 \
```

### fix find_package not found
```
sudo apt install libxxf86vm-dev
sudo apt install libudev-dev
```

### build now
```
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH

export BUILD_PYTORCH_OPS=ON
export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_TENSORFLOW_OPS=OFF
export CLOUDVIEWER_ML_ROOT=/home/asher/develop/code/github/CloudViewer/CloudViewer-ML

cd ACloudViewer
mkdir build
cd build
cmake -DDEVELOPER_BUILD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBREALSENSE=ON \
      -DBUILD_AZURE_KINECT=ON \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_OPENCV=OFF \
      -DWITH_OPENMP=ON \
      -DWITH_IPP=ON \
      -DWITH_SIMD=ON \
      -DUSE_QT6=OFF \
      -DUSE_SIMD=ON \
      -DCVCORELIB_SHARED=ON \
      -DCVCORELIB_USE_CGAL=ON \
      -DCVCORELIB_USE_QT_CONCURRENT=ON \
      -DBUILD_WEBRTC=ON \
      -DUSE_PCL_BACKEND=OFF \
      -DBUILD_FILAMENT_FROM_SOURCE=OFF \
      -DBUILD_JUPYTER_EXTENSION=ON \
      -DBUILD_RECONSTRUCTION=ON \
      -DBUILD_COMMON_CUDA_ARCHS=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_CLOUDVIEWER_ML=ON \
      -DBUILD_WITH_CONDA=ON \
      -DCONDA_PREFIX=$CONDA_PREFIX \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
      -DCMAKE_INSTALL_PREFIX=/home/asher/develop/code/github/CloudViewer/install \
      -DCLOUDVIEWER_ML_ROOT=/home/asher/develop/code/github/CloudViewer/CloudViewer-ML \
      ..

make "-j$(nproc)" python-package

# build with cuda
cmake -DBUILD_CUDA_MODULE=ON ..

make "-j$(nproc)" pip-package
make "-j$(nproc)" install-pip-package
python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"

```

## Test
```
cd ${CLOUDVIEWER_SOURCE_ROOT}
export BUILD_PYTORCH_OPS=ON
export BUILD_TENSORFLOW_OPS=OFF
export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export CLOUDVIEWER_ML_ROOT=/home/asher/develop/code/github/CloudViewer/CloudViewer-ML
source util/ci_utils.sh

test_wheel build/lib/python_package/pip_package/cloudviewer*

# test c++ and python
run_all_tests

# test c++
run_cpp_unit_tests

# test python
run_python_tests
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

