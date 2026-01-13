.. _compilation:

Building from Source
====================

This guide covers how to build ACloudViewer from source on Linux, macOS, and Windows.

.. note::
   ACloudViewer is a comprehensive 3D point cloud processing application with many optional features.
   The build process is complex but well-documented below.

Quick Start
-----------

**Linux (Docker - Recommended):**

.. code-block:: bash

   ./docker/build-release.sh
   # Or with conda:
   ./docker/build-release-conda.sh

**Linux (Manual):**

.. code-block:: bash

   utils/install_deps_ubuntu.sh assume-yes
   ./scripts/build_ubuntu.sh

**macOS:**

.. code-block:: bash

   ./scripts/build_macos.sh

**Windows:**

.. code-block:: powershell

   python .\scripts\build_win.py

System Requirements
-------------------

Minimum Requirements
~~~~~~~~~~~~~~~~~~~~

- **OS**: Ubuntu 18.04+, macOS 10.15+, Windows 10+
- **CPU**: 8+ cores recommended
- **RAM**: 16 GB+ recommended
- **Disk**: 20 GB free space
- **CMake**: 3.19 or higher

Recommended
~~~~~~~~~~~

- **CPU**: 16+ cores
- **RAM**: 32 GB
- **GPU**: NVIDIA GPU with CUDA 11.8+ (Linux/Windows only)
- **Internet**: Good connection for downloading dependencies

Build Options
-------------

ACloudViewer supports numerous build options:

Core Options
~~~~~~~~~~~~

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Option
     - Description
   * - ``DEVELOPER_BUILD``
     - Enable developer mode (OFF for release builds)
   * - ``CMAKE_BUILD_TYPE``
     - Build type: Debug, Release, RelWithDebInfo
   * - ``BUILD_SHARED_LIBS``
     - Build shared libraries (OFF for Windows)
   * - ``PACKAGE``
     - Enable packaging/installation support

Feature Options
~~~~~~~~~~~~~~~

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Option
     - Description
   * - ``BUILD_CUDA_MODULE``
     - Enable CUDA support (Linux/Windows only)
   * - ``BUILD_OPENCV``
     - Build with OpenCV support
   * - ``BUILD_RECONSTRUCTION``
     - Enable 3D reconstruction features
   * - ``BUILD_PYTHON_MODULE``
     - Build Python bindings
   * - ``USE_PCL_BACKEND``
     - Use PCL as processing backend
   * - ``WITH_OPENMP``
     - Enable OpenMP parallel processing
   * - ``WITH_SIMD``
     - Enable SIMD optimizations

Plugin Options
~~~~~~~~~~~~~~

ACloudViewer has a rich plugin system. Key plugins:

- **I/O Plugins**: QDRACO, QLAS, QE57, QMESH, QPHOTOSCAN, QRDB, QFBX
- **Standard Plugins**: QCORK, QANIMATION, QCANUPO, QCSF, QM3C2, QPCL, QPOISSON_RECON
- **Masonry Plugins**: QAUTO_SEG, QMANUAL_SEG

See platform-specific sections for complete plugin lists.

Linux (Ubuntu/Debian)
----------------------

System Dependencies
~~~~~~~~~~~~~~~~~~~

Install dependencies automatically:

.. code-block:: bash

   # Install all dependencies
   utils/install_deps_ubuntu.sh assume-yes

Or install manually:

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y \
       build-essential cmake git pkg-config \
       libeigen3-dev libflann-dev libboost-all-dev \
       qt5-default libqt5svg5-dev libqt5opengl5-dev qttools5-dev \
       libglu1-mesa-dev freeglut3-dev mesa-common-dev

Python Environment (pyenv)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export PYENV_ROOT=~/.pyenv PYTHON_VERSION=3.12
   export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH"
   
   curl https://pyenv.run | bash
   pyenv install $PYTHON_VERSION
   pyenv global $PYTHON_VERSION
   pyenv rehash
   
   python --version && pip --version

Building Application (Full Configuration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
   export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH
   
   cd ACloudViewer
   mkdir build_app && cd build_app
   
   cmake \
       -DDEVELOPER_BUILD=OFF \
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
       -DPLUGIN_IO_QRDB=ON \
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
       -DPLUGIN_STANDARD_G3POINT=ON \
       -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON \
       -DPLUGIN_STANDARD_QM3C2=ON \
       -DPLUGIN_STANDARD_QMPLANE=ON \
       -DPLUGIN_STANDARD_QPCL=ON \
       -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
       -DPOISSON_RECON_WITH_OPEN_MP=ON \
       -DPLUGIN_STANDARD_QSRA=ON \
       -DPLUGIN_STANDARD_3DMASC=ON \
       -DPLUGIN_STANDARD_QTREEISO=ON \
       -DPLUGIN_STANDARD_QVOXFALL=ON \
       -DPLUGIN_PYTHON=ON \
       -DBUILD_PYTHON_MODULE=ON \
       -DBUILD_UNIT_TESTS=OFF \
       ..
   
   make -j$(nproc)
   make install

.. note::
   - Use ``DUSE_QT6=ON`` for Ubuntu 24.04+
   - ``BUILD_CUDA_MODULE=ON`` requires NVIDIA GPU and CUDA toolkit
   - Adjust ``-j`` value based on available CPU cores and RAM

Building Python Wheel
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd ACloudViewer
   mkdir build_wheel && cd build_wheel
   
   cmake \
       -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_SHARED_LIBS=OFF \
       -DBUILD_PYTHON_MODULE=ON \
       -DBUILD_CUDA_MODULE=ON \
       -DSTATIC_WINDOWS_RUNTIME=ON \
       ..
   
   make -j$(nproc) pip-package
   
   # Install wheel
   pip install lib/python/pip-package/*.whl

Docker Build (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fastest and most reliable method:

.. code-block:: bash

   # Standard build
   ./docker/build-release.sh
   
   # Or with conda environment
   ./docker/build-release-conda.sh

Debugging Python Wheel
~~~~~~~~~~~~~~~~~~~~~~~

If import fails:

.. code-block:: bash

   # Method 1: Using gdb batch mode
   gdb --batch --ex run --ex bt --ex quit --args python3 -c "import cloudViewer"
   
   # Method 2: Interactive gdb
   gdb python3
   # In gdb:
   # (gdb) run -c "import cloudViewer"
   # (gdb) bt

macOS
-----

System Dependencies
~~~~~~~~~~~~~~~~~~~

Install Xcode and Homebrew:

.. code-block:: bash

   xcode-select --install
   
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   
   brew install gcc --without-multilib

Python Environment (conda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cp .ci/conda_macos_cloudViewer.yml /tmp/conda_macos_cloudViewer.yml
   sed -i "" "s/3.8/3.12/g" /tmp/conda_macos_cloudViewer.yml
   conda env create -f /tmp/conda_macos_cloudViewer.yml
   conda activate cloudViewer

Building Application
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
   export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH
   
   cd ACloudViewer
   mkdir build_app && cd build_app
   
   cmake \
       -DDEVELOPER_BUILD=OFF \
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
       -DPLUGIN_STANDARD_G3POINT=ON \
       -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON \
       -DPLUGIN_STANDARD_QM3C2=ON \
       -DPLUGIN_STANDARD_QMPLANE=ON \
       -DPLUGIN_STANDARD_QPCL=ON \
       -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
       -DPOISSON_RECON_WITH_OPEN_MP=ON \
       -DPLUGIN_STANDARD_QSRA=ON \
       -DPLUGIN_STANDARD_3DMASC=ON \
       -DPLUGIN_STANDARD_QTREEISO=ON \
       -DPLUGIN_STANDARD_QVOXFALL=ON \
       -DPLUGIN_PYTHON=ON \
       -DBUILD_PYTHON_MODULE=ON \
       -DBUILD_UNIT_TESTS=OFF \
       ..
   
   make -j$(sysctl -n hw.ncpu)

.. warning::
   **macOS Limitations:**
   
   - CUDA is **NOT supported** on macOS (Apple removed NVIDIA GPU support)
   - OpenCV may have issues on some machines
   - Some plugins may not be available

Building Python Wheel
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   cd ACloudViewer
   mkdir build_wheel && cd build_wheel
   
   cmake \
       -DCMAKE_BUILD_TYPE=Release \
       -DBUILD_SHARED_LIBS=OFF \
       -DBUILD_PYTHON_MODULE=ON \
       -DBUILD_CUDA_MODULE=OFF \
       ..
   
   make -j$(sysctl -n hw.ncpu) pip-package
   pip install lib/python/pip-package/*.whl

Universal Binary (Intel + ARM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For distribution on both Intel and Apple Silicon Macs:

.. code-block:: bash

   cmake \
       -DCMAKE_OSX_ARCHITECTURES="arm64;x86_64" \
       -DCMAKE_BUILD_TYPE=Release \
       ...

Debugging Python Wheel
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   lldb python
   # In lldb:
   # (lldb) run -c "import cloudViewer"
   # (lldb) bt

Windows
-------

System Dependencies
~~~~~~~~~~~~~~~~~~~

1. **Visual Studio 2019 or 2022**
   
   Download from: https://visualstudio.microsoft.com/
   
   Required components:
   
   - Desktop development with C++
   - CMake tools
   - Windows 10/11 SDK

2. **CMake** (3.19+): https://cmake.org/download/

3. **Git**: https://git-scm.com/download/win

Python Environment (conda)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   $env:CLOUDVIEWER_SOURCE_ROOT = (Get-Location).Path
   Copy-Item (Join-Path $env:CLOUDVIEWER_SOURCE_ROOT ".ci\conda_windows_cloudViewer.yml") -Destination "$env:TEMP\conda_windows_cloudViewer.yml"
   (Get-Content "$env:TEMP\conda_windows_cloudViewer.yml") -replace "3.8", "3.12" | Set-Content "$env:TEMP\conda_windows_cloudViewer.yml"
   
   conda env create -f "$env:TEMP\conda_windows_cloudViewer.yml"
   conda activate cloudViewer
   
   $env:GENERATOR = "Visual Studio 17 2022"
   $env:ARCHITECTURE = "x64"
   $env:NPROC = (Get-CimInstance -ClassName Win32_ComputerSystem).NumberOfLogicalProcessors
   $env:CLOUDVIEWER_INSTALL_DIR = "C:/dev/cloudViewer_install"

Building Application
~~~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   mkdir build_app
   cd build_app
   conda activate cloudViewer
   ..\scripts\setup_conda_env.ps1
   
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
       -DCMAKE_INSTALL_PREFIX="$env:CLOUDVIEWER_INSTALL_DIR" `
       ..
   
   cmake --build . --config Release --parallel $env:NPROC
   cmake --install . --prefix $env:CLOUDVIEWER_INSTALL_DIR

Building Python Wheel
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   mkdir build_wheel
   cd build_wheel
   conda activate cloudViewer
   
   cmake -G $env:GENERATOR -A $env:ARCHITECTURE `
       -DCMAKE_BUILD_TYPE=Release `
       -DBUILD_SHARED_LIBS=OFF `
       -DSTATIC_WINDOWS_RUNTIME=ON `
       -DBUILD_PYTHON_MODULE=ON `
       -DBUILD_CUDA_MODULE=ON `
       ...
   
   cmake --build . --target pip-package --config Release
   pip install lib\python\pip-package\*.whl

Using Build Script (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: powershell

   python .\scripts\build_win.py

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Out of Memory During Build**

.. code-block:: bash

   # Reduce parallel jobs
   make -j4  # Instead of -j$(nproc)

**CUDA Not Found**

.. code-block:: bash

   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

**Qt Not Found**

.. code-block:: bash

   export CMAKE_PREFIX_PATH="/path/to/qt/6.5.3/gcc_64"

**Plugin Build Failures**

Disable problematic plugins:

.. code-block:: bash

   cmake -DPLUGIN_STANDARD_QFBX=OFF ...

**Submodule Issues**

.. code-block:: bash

   git submodule update --init --recursive --force

Platform-Specific Notes
-----------------------

Linux
~~~~~

- Use ``ccache`` for faster rebuilds
- Consider using Docker for reproducible builds
- CUDA support requires NVIDIA GPU

macOS
~~~~~

- **CUDA NOT supported**
- Universal binaries increase build time 2x
- Some OpenCV features may not work
- Use Homebrew for dependencies

Windows
~~~~~~~

- Use PowerShell (not CMD)
- Visual Studio 2019+ required
- Build times are longer than Linux
- Use ``build_win.py`` script for easier setup

Build Performance Tips
----------------------

1. **Use SSD** for build directory
2. **Increase RAM** (32GB+ recommended)
3. **Use ccache** (Linux/macOS)
4. **Reduce parallel jobs** if running out of memory
5. **Disable unused plugins** to save time
6. **Use Docker** for clean, reproducible builds

Next Steps
----------

After building:

1. :doc:`quickstart` - Learn basic usage
2. :doc:`../tutorial/index` - Follow tutorials
3. :doc:`../examples/cpp_examples` - C++ examples
4. :doc:`../examples/python_examples` - Python examples

Additional Resources
--------------------

**Detailed Build Guides:**

- `Linux Build Guide (Markdown) <https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/compiling_doc/compiling-cloudviewer-linux.md>`_
- `macOS Build Guide (Markdown) <https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/compiling_doc/compiling-cloudviewer-macos.md>`_
- `Windows Build Guide (Markdown) <https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/compiling_doc/compiling-cloudviewer-windows.md>`_

**Build Scripts:**

- Linux: ``scripts/build_ubuntu.sh``
- macOS: ``scripts/build_macos.sh``
- Windows: ``scripts/build_win.py``
- Docker: ``docker/build-release.sh``

**Related Documentation:**

- :doc:`../developer/contributing` - Contributing guidelines
- :doc:`../developer/docker` - Docker usage
- `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_ - Report problems

.. note::
   This documentation is based on the actual build scripts and configuration files in the repository.
   For the most up-to-date information, refer to the Markdown files in ``docs/guides/compiling_doc/``.
