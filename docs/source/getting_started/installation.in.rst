Installation
============

Pre-built Binaries
------------------

The easiest way to install ACloudViewer is using pre-built packages.

Python Installation
~~~~~~~~~~~~~~~~~~~

Download wheel files from `GitHub Releases <https://github.com/Asher-1/ACloudViewer/releases>`_.

.. note::
   
   **Current version:** |cv_version|
   
   Download from the `Releases page <https://github.com/Asher-1/ACloudViewer/releases>`_.

**Linux (Ubuntu 22.04, Python 3.13, CUDA)**

.. code-block:: bash

   wget https://github.com/Asher-1/ACloudViewer/releases/download/v@cv_version@/cloudviewer-@cv_version@-cp313-cp313-manylinux_2_35_x86_64.whl
   pip install cloudviewer-@cv_version@-cp313-cp313-manylinux_2_35_x86_64.whl

**Linux (Ubuntu 22.04, Python 3.13, CPU-only)**

.. code-block:: bash

   wget https://github.com/Asher-1/ACloudViewer/releases/download/v@cv_version@/cloudviewer_cpu-@cv_version@-cp313-cp313-manylinux_2_35_x86_64.whl
   pip install cloudviewer_cpu-@cv_version@-cp313-cp313-manylinux_2_35_x86_64.whl

**macOS (ARM64, Python 3.12)**

.. code-block:: bash

   wget https://github.com/Asher-1/ACloudViewer/releases/download/v@cv_version@/cloudViewer-@cv_version@-cp312-cp312-macosx_11_0_arm64.whl
   pip install cloudViewer-@cv_version@-cp312-cp312-macosx_11_0_arm64.whl

**Windows (Python 3.12, CUDA)**

.. code-block:: powershell

   # Download from GitHub Releases
   pip install cloudViewer-@cv_version@-cp312-cp312-win_amd64.whl

.. tip::
   
   Adjust ``cp312`` or ``cp313`` for your Python version (cp310 for 3.10, cp311 for 3.11, cp312 for 3.12, cp313 for 3.13).

Desktop Application
~~~~~~~~~~~~~~~~~~~

Download installers for your platform from `GitHub Releases <https://github.com/Asher-1/ACloudViewer/releases>`_.

**Linux (Ubuntu 22.04+)**

.. code-block:: bash

   # Download the .run installer
   wget https://github.com/Asher-1/ACloudViewer/releases/download/v@cv_version@/ACloudViewer-@cv_version@-ubuntu22.04-cuda-amd64.run
   chmod +x ACloudViewer-@cv_version@-ubuntu22.04-cuda-amd64.run
   ./ACloudViewer-@cv_version@-ubuntu22.04-cuda-amd64.run

**macOS (11.0+)**

.. code-block:: bash

   # Download and open DMG file
   open ACloudViewer-@cv_version@-mac-cpu-ARM64.dmg

Or drag and drop the application to Applications folder.

**Windows (10+)**

Download the ``.exe`` installer from the releases page and run it. Follow the installation wizard.

Via npm (cross-platform bootstrap)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Instead of downloading an installer manually, the ``@asher-1/acloudviewer`` npm
package fetches and launches the matching official installer for your platform
(Windows / macOS / Linux) automatically. The package version mirrors the app
version, and three release channels are available:

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Channel
     - Install command
     - What you get
   * - **Stable** (default)
     - ``npm install -g @asher-1/acloudviewer``
     - Latest official release (``latest`` dist-tag = app version); pins the
       matching ``v<version>`` release.
   * - **Preview / testing**
     - ``npm install -g @asher-1/acloudviewer@next``
     - Newest complete ``main-devel`` build (``X.Y.Z-dev.<timestamp>``); tracks
       the rolling development assets.
   * - **Exact version**
     - ``npm install -g @asher-1/acloudviewer@``\ @cv_version@
     - A specific published bootstrap version.

.. code-block:: bash

   npm install -g @asher-1/acloudviewer   # or @next for the preview channel
   acloudviewer install                   # CPU by default; --cuda opts in (driver checked)
   acloudviewer list                      # show available installers
   acloudviewer uninstall --yes           # remove (QtIFW maintenancetool purge)

``latest`` only ever points at a stable release; development builds are published
under ``next`` and never move it. See ``npm/PUBLISHING.md`` in the repository for
the full version scheme and the transitional ``3.9.5`` semantics.

Docker Installation
-------------------

Docker is used for **reproducible builds and headless environments**, not for
distributing the desktop installers: the release artifacts (``.run`` /
``.exe`` / ``.dmg``) are GUI desktop packages, while containers are headless
by nature. No prebuilt application image is published; the Docker images in
this repository are build environments that compile the installers and
wheels from source:

.. code-block:: bash

   ./docker/build-release.sh    # build the release installers inside Docker

For headless / container usage, install the Python wheel from the GitHub
release assets (see the Python section above) or the CLI harness — both run
without a display server.

See :doc:`../developer/docker` for more information.

Verify Installation
-------------------

Python
~~~~~~

.. code-block:: python

   import cloudViewer as cv3d
   print(cv3d.__version__)
   
   # Test basic functionality
   pcd = cv3d.geometry.ccPointCloud()
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
* :doc:`../tutorial/index` - Tutorial index

