Build from source in Ubuntu and macOS
=====================

1. Install dependencies

	    # On Ubuntu
	    scripts/install_deps_ubuntu.sh assume-yes
	
	    # On macOS
	    # Install Homebrew first: https://brew.sh/
	    scripts/install_deps_macos.sh
	
	    # configure for vtk(8.2)
	      cmake -DVTK_QT_VERSION:STRING=5 \
		-DCMAKE_BUILD_TYPE=Release \
	      	-DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake \
	      	-DVTK_Group_Qt:BOOL=ON \
	      	-DCMAKE_PREFIX_PATH:PATH=/opt/5.14.2/5.14.2/gcc_64/lib/cmake  \
	      	-DBUILD_SHARED_LIBS:BOOL=ON ..
	
		make -j 8
		sudo make install
	    # cofigure for qt VTK PLUGINS
		sudo find / -name libQVTKWidgetPlugin.so
		sudo cp lib/libQVTKWidgetPlugin.so /opt/Qt5.14.2/5.14.2/gcc_64/plugins/designer
		sudo cp lib/libQVTKWidgetPlugin.so /opt/Qt5.14.2/Tools/QtCreator/lib/Qt/plugins/designer
	
	    # cofigure PCL(1.11.1)
		cmake -DCMAKE_BUILD_TYPE=Release \
		      -DBUILD_GPU=ON \
		      -DBUILD_apps=ON \
		      -DBUILD_examples=ON \
		      -DBUILD_surface_on_nurbs=ON \
		      -DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake \
		      -DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake ..
	
		make -j 8
		sudo make install

2. Setup Python environments

Activate the python ``virtualenv`` or Conda ``virtualenv```. Check
``which python`` to ensure that it shows the desired Python executable.
Alternatively, set the CMake flag ``-DPYTHON_EXECUTABLE=/path/to/python``
to specify the python executable.
If Python binding is not needed, you can turn it off by ``-DBUILD_PYTHON_MODULE=OFF``.

3. Config

	    mkdir build
	    cd build
	    cmake -DCMAKE_BUILD_TYPE=Release \
	      	-DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.13.0/5.13.0/gcc_64/bin/qmake \
	      	-DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.13.0/5.13.0/gcc_64/lib/cmake  \
	      	../ACloudViewer
	
	    cmake -DCMAKE_BUILD_TYPE=Release \
	      	-DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake \
	      	-DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake  \
	      	-DCMAKE_INSTALL_PREFIX=<cloudViewer_install_directory> ..

The ``CMAKE_INSTALL_PREFIX`` argument is optional and can be used to install
CloudViewer to a user location. In the absence of this argument CloudViewer will be
installed to a system location where ``sudo`` is required) For more
options of the build, see :ref:`compilation_options`.


4. Build

	    # On Ubuntu
	    make -j$(nproc)
	
	    # On macOS
	    make -j$(sysctl -n hw.physicalcpu)


5. Install

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


Compilation options
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


Unit test
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


Package Linux
-------------

[package-on-linux.md file](package-on-linux.md)
