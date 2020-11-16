Build from source
=====================

System requirements
-------------------

* Ubuntu 18.04+: GCC 5+, Clang 7+
* macOS 10.14+: XCode 8.0+
* Windows 10 (64-bit): Visual Studio 2019+
* CMake: 3.15+ for Ubuntu and macOS, 3.18+ for Windows

  * Ubuntu (18.04):

    * Install with ``apt-get``: see `official APT repository <https://apt.kitware.com/>`_
    * Install with ``snap``: ``sudo snap install cmake --classic``
    * Install with ``pip`` (run inside a Python virtualenv): ``pip install cmake``

  * Ubuntu (20.04+): Use the default OS repository: ``sudo apt-get install cmake``
  * macOS: Install with Homebrew: ``brew install cmake``
  * Windows: Download from: `CMake download page <https://cmake.org/download/>`_

* CUDA 10.1 (optional): CloudViewer supports GPU acceleration of an increasing number
  of operations through CUDA on Linux. Please see the `official documentation
  <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_ to
  install the CUDA toolkit from Nvidia.


Cloning CloudViewer
--------------

Make sure to use the ``--recursive`` flag when cloning CloudViewer.

.. code-block:: bash

    git clone --recursive https://github.com/Asher-1/ErowCloudViewer.git

    # You can also update the submodule manually
    git submodule update --init --recursive



Ubuntu/macOS
------------

Refer to the [compiling-cloudviewer-linux.md file](compiling-cloudviewer-linux.md) for compilation Ubuntu/macOS information.


Windows
-------

1. Setup Python binding environments
````````````````````````````````````

Most steps are the steps for Ubuntu: :ref:`compilation_unix_python`.
Instead of ``which``, check the Python path with ``where python``


2. Config
`````````

    mkdir build
    cd build

    :: Specify the generator based on your Visual Studio version
    :: If CMAKE_INSTALL_PREFIX is a system folder, admin access is needed for installation
    cmake -G "Visual Studio 16 2019" -A x64 -DCMAKE_INSTALL_PREFIX="<cloudViewer_install_directory>" ..

3. Build
````````

.. code-block:: bat

    cmake --build . --config Release --target ALL_BUILD

Alternatively, you can open the ``CloudViewer.sln`` project with Visual Studio and
build the same target.

4. Install
``````````

To install CloudViewer C++ library, build the ``INSTALL`` target in terminal or
in Visual Studio.

    cmake --build . --config Release --target INSTALL

To link a C++ project against the CloudViewer C++ library, please refer to
:ref:`create_cplusplus_project`.

To install CloudViewer Python library, build the corresponding python installation
targets in terminal or Visual Studio.

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

Compilation options
-------------------

OpenMP
``````

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

.. note:: This workaround has some compatibility issues with the source code of
    GLFW included in ``3rdparty``.
    Make sure CloudViewer is linked against GLFW installed on the OS.

ML Module
`````````

The ML module consists of primitives like operators and layers as well as high
level code for models and pipelines. To build the operators and layers, set
``BUILD_PYTORCH_OPS=ON`` and/or ``BUILD_TENSORFLOW_OPS=ON``.  Don't forget to also
enable ``BUILD_CUDA_MODULE=ON`` for GPU support. To include the models and
pipelines from CloudViewer-ML in the python package, set ``BUNDLE_OPEN3D_ML=ON`` and
``OPEN3D_ML_ROOT`` to the CloudViewer-ML repository. You can directly download
CloudViewer-ML from GitHub during the build with
``OPEN3D_ML_ROOT=https://github.com/intel-isl/CloudViewer-ML.git``.

The following example shows the command for building the ops with GPU support
for all supported ML frameworks and bundling the high level CloudViewer-ML code.

.. code-block:: bash

    # In the build directory
    cmake -DBUILD_CUDA_MODULE=ON \
          -DBUILD_PYTORCH_OPS=ON \
          -DBUILD_TENSORFLOW_OPS=ON \
          -DBUNDLE_OPEN3D_ML=ON \
          -DOPEN3D_ML_ROOT=https://github.com/intel-isl/CloudViewer-ML.git \
          ..
    # Install the python wheel with pip
    make -j install-pip-package

.. note::
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

       .. code-block:: bash

           python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
           python -c "import tensorflow; print(tensorflow.__cxx11_abi_flag__)"

    2. Configure CloudViewer to compile all dependencies from source
       with the corresponding ABI version obtained from step 1.

    After installation of the Python package, you can check CloudViewer ABI version
    with:

    .. code-block:: bash

        python -c "import cloudViewer; print(cloudViewer.pybind._GLIBCXX_USE_CXX11_ABI)"

    To build CloudViewer with CUDA support, configure with:

    .. code-block:: bash

        cmake -DBUILD_CUDA_MODULE=ON -DCMAKE_INSTALL_PREFIX=<cloudViewer_install_directory> ..

    Please note that CUDA support is work in progress and experimental. For building
    CloudViewer with CUDA support, ensure that CUDA is properly installed by running following commands:

    .. code-block:: bash

        nvidia-smi      # Prints CUDA-enabled GPU information
        nvcc -V         # Prints compiler version

    If you see an output similar to ``command not found``, you can install CUDA toolkit
    by following the `official
    documentation. <https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html>`_


Unit test
---------

To build and run C++ unit tests:

.. code-block:: bash

    cmake -DBUILD_UNIT_TESTS=ON ..
    make -j
    ./bin/tests


To run Python unit tests:

.. code-block:: bash

    # Activate virtualenv first
    pip install pytest
    make install-pip-package
    pytest ../python/test
