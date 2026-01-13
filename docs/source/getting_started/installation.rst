Installation
============

Pre-built Binaries
------------------

The easiest way to install ACloudViewer is using pre-built packages.

Python Installation
~~~~~~~~~~~~~~~~~~~

Download wheel files from `GitHub Releases <https://github.com/Asher-1/ACloudViewer/releases>`_.

**Linux (Ubuntu 22.04, Python 3.10, CUDA)**

.. code-block:: bash

   wget https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.3/cloudviewer-3.9.3-cp310-cp310-manylinux_2_35_x86_64.whl
   pip install cloudviewer-3.9.3-cp310-cp310-manylinux_2_35_x86_64.whl

**Linux (Ubuntu 22.04, Python 3.10, CPU-only)**

.. code-block:: bash

   wget https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.3/cloudviewer_cpu-3.9.3-cp310-cp310-manylinux_2_35_x86_64.whl
   pip install cloudviewer_cpu-3.9.3-cp310-cp310-manylinux_2_35_x86_64.whl

**macOS (ARM64, Python 3.10)**

.. code-block:: bash

   wget https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.3/cloudViewer-3.9.3-cp310-cp310-macosx_11_0_arm64.whl
   pip install cloudViewer-3.9.3-cp310-cp310-macosx_11_0_arm64.whl

**Windows (Python 3.10, CUDA)**

.. code-block:: powershell

   # Download from GitHub Releases
   pip install cloudViewer-3.9.3-cp310-cp310-win_amd64.whl

Desktop Application
~~~~~~~~~~~~~~~~~~~

Download installers from `GitHub Releases <https://github.com/Asher-1/ACloudViewer/releases>`_.

**Linux**

.. code-block:: bash

   # Ubuntu 22.04 with CUDA
   wget https://github.com/Asher-1/ACloudViewer/releases/download/v3.9.3/ACloudViewer-3.9.3-ubuntu22.04-cuda-amd64.run
   chmod +x ACloudViewer-3.9.3-ubuntu22.04-cuda-amd64.run
   ./ACloudViewer-3.9.3-ubuntu22.04-cuda-amd64.run

**macOS**

.. code-block:: bash

   # Download and open DMG file
   open ACloudViewer-3.9.3-mac-cpu-ARM64.dmg

**Windows**

Download and run the EXE installer.

Docker Installation
-------------------

Docker images are available for easy deployment.

.. code-block:: bash

   docker pull ghcr.io/asher-1/acloudviewer:latest
   docker run -it --gpus all ghcr.io/asher-1/acloudviewer:latest

See :doc:`../developer/docker` for more information.

Verify Installation
-------------------

Python
~~~~~~

.. code-block:: python

   import cloudViewer as cv3d
   print(cv3d.__version__)
   
   # Test basic functionality
   pcd = cv3d.geometry.PointCloud()
   print("Installation successful!")

C++
~~~

.. code-block:: cpp

   #include <cloudViewer/CloudViewer.h>
   #include <iostream>
   
   int main() {
       std::cout << "CloudViewer version: " 
                 << cv::utility::GetVersion() << std::endl;
       return 0;
   }

Troubleshooting
---------------

CUDA Issues
~~~~~~~~~~~

If you encounter CUDA-related errors:

1. Check CUDA installation:

.. code-block:: bash

   nvidia-smi
   nvcc --version

2. Ensure CUDA version matches (11.8 or 12.x)

3. Install CPU-only version if GPU is not required

Import Errors (Python)
~~~~~~~~~~~~~~~~~~~~~~

If you get ``ImportError``:

1. Check Python version (3.8-3.12 supported)
2. Ensure correct wheel file for your system
3. Install in a clean virtual environment

.. code-block:: bash

   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install cloudviewer-*.whl

Library Not Found (Linux)
~~~~~~~~~~~~~~~~~~~~~~~~~~

If you get ``library not found`` errors:

.. code-block:: bash

   export LD_LIBRARY_PATH=/path/to/cloudviewer/lib:$LD_LIBRARY_PATH

Next Steps
----------

* :doc:`quickstart` - Quick start guide
* :doc:`build_from_source` - Build from source
* :doc:`../tutorials/basic_usage` - Basic usage tutorial

