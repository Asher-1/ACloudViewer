Frequently Asked Questions
===========================

General Questions
-----------------

What is ACloudViewer?
^^^^^^^^^^^^^^^^^^^^^

ACloudViewer is a comprehensive C++ library for 3D point cloud and mesh processing. It provides:

- Point cloud visualization and processing
- Mesh manipulation and analysis
- 3D reconstruction algorithms
- Machine learning integration
- Cross-platform GUI application
- Python and C++ APIs

How is ACloudViewer different from other 3D libraries?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ACloudViewer combines:

- **Performance**: GPU-accelerated operations with CUDA support
- **Completeness**: Integrated visualization, processing, and ML capabilities
- **Flexibility**: Both GUI application and programming libraries
- **Modern**: Built with modern C++17 and Python 3.10+
- **Extensibility**: Rich plugin system for custom features

Installation & Setup
--------------------

Which platforms are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- **Linux**: Ubuntu 20.04+, other distributions
- **macOS**: macOS 10.15+ (Intel and Apple Silicon)
- **Windows**: Windows 10+ (64-bit)

What are the system requirements?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Minimum:**

- CPU: 64-bit x86 processor
- RAM: 4 GB
- Disk: 2 GB free space
- Graphics: OpenGL 3.3+ support

**Recommended:**

- CPU: Multi-core 64-bit processor
- RAM: 16 GB or more
- Disk: 10 GB free space (SSD)
- Graphics: NVIDIA GPU with CUDA support
- CUDA: 12.x for GPU acceleration

How do I install ACloudViewer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**GUI Application:**

Download the installer from the `Downloads page <index.html#download>`_.

**Python Package:**

.. code-block:: bash

   # Download .whl from GitHub Releases
   pip install cloudviewer-x.x.x-cpXX-cpXX-platform.whl

**From Source:**

See :doc:`/getting_started/build_from_source`.

Do I need CUDA for ACloudViewer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

No, CUDA is optional. ACloudViewer works without GPU acceleration, but CUDA significantly improves performance for:

- Large point cloud processing
- 3D reconstruction
- Neural network operations

Usage Questions
---------------

Can I use ACloudViewer commercially?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, ACloudViewer is released under a permissive license that allows commercial use. See the ``LICENSE`` file for details.

What file formats are supported?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ACloudViewer supports a wide range of file formats through core I/O libraries (``libs/eCV_io``) and extensible I/O plugins (``plugins/core/IO``).

**Point Cloud Formats:**

- **PCD** - Point Cloud Data (PCL format)
- **PLY** - Polygon File Format (ASCII/Binary)
- **LAS/LAZ** - ASPRS LiDAR formats (via qLASIO, qLASFWFIO, qPDALIO plugins)
- **E57** - ASTM E57 format (via qE57IO plugin)
- **XYZ/TXT/ASC** - ASCII point clouds
- **CSV** - Comma-separated values (via qCSVMatrixIO plugin)
- **PTS** - Leica point cloud format
- **PTX** - Leica PTX format
- **RDB** - Riegl database (via qRDBIO plugin)
- **Photoscan** - Agisoft Photoscan (via qPhotoscanIO plugin)
- **Draco** - Google Draco compressed (via qDracoIO plugin)

**Mesh Formats:**

- **PLY** - Polygon File Format (with textures)
- **OBJ/MTL** - Wavefront OBJ (with materials and textures)
- **STL** - Stereolithography (ASCII/Binary)
- **FBX** - Autodesk FBX (via qFBXIO plugin)
- **GLTF/GLB** - GL Transmission Format (via qMeshIO plugin)
- **OFF** - Object File Format
- **3DS** - 3D Studio format
- **STEP/STP** - STEP CAD format (via qStepCADImport plugin)
- **MA** - Maya ASCII
- **VTK** - VTK formats

**Image Formats:**

- **PNG** - Portable Network Graphics
- **JPEG/JPG** - Joint Photographic Experts Group
- **BMP** - Bitmap
- **TIFF/TIF** - Tagged Image File Format
- **TGA** - Truevision TGA
- **HDR** - High Dynamic Range
- **EXR** - OpenEXR

**Additional Formats (via qAdditionalIO & qCoreIO plugins):**

- **BIN** - ACloudViewer binary format
- **SHP** - ESRI Shapefile
- **DXF** - AutoCAD Drawing Exchange Format
- **POV/ICM/PN/PV** - Various specialized formats
- **SOI** - Mensi Soisic cloud
- **BUNDLER** - Snavely's Bundler output
- **ASCII matrices** - Generic ASCII data

**Format Notes:**

- Most formats support both import and export
- Plugin-based formats require corresponding plugins to be enabled
- Some formats (LAZ, FBX, STEP) may have platform-specific availability
- For full format details, see the `File I/O documentation <../python_api/cloudViewer.io.html>`_

How do I load large point clouds efficiently?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import cloudViewer as cv3d
   
   # Use chunked loading for large files
   pcd = cv3d.io.read_point_cloud("large_file.las", 
                                   print_progress=True)
   
   # Or use LOD (Level of Detail)
   pcd = pcd.voxel_down_sample(voxel_size=0.05)

Can I process point clouds in batches?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes, use multiprocessing or the batch processing API:

.. code-block:: python

   from cloudViewer import geometry
   from concurrent.futures import ThreadPoolExecutor
   
   def process_cloud(filename):
       pcd = geometry.PointCloud.read(filename)
       # Your processing here
       return pcd
   
   with ThreadPoolExecutor(max_workers=4) as executor:
       results = executor.map(process_cloud, file_list)

Development Questions
---------------------

How do I build from source?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See the comprehensive guide: :doc:`/getting_started/build_from_source`.

**Quick Start:**

.. code-block:: bash

   git clone https://github.com/Asher-1/ACloudViewer.git
   cd ACloudViewer
   mkdir build && cd build
   cmake ..
   make -j$(nproc)

What are the build dependencies?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Core Dependencies:**

- CMake 3.20+
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- Python 3.10+ (for Python bindings)
- Qt5/Qt6 (for GUI)

**Optional Dependencies:**

- CUDA 12.x (for GPU acceleration)
- PCL (for additional algorithms)
- OpenCV (for image processing)

See :doc:`/getting_started/build_from_source` for complete list.

How do I contribute to ACloudViewer?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :doc:`/developer/contributing` for the complete guide.

**Quick Steps:**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

Python API Questions
--------------------

Why is the Python package so large?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Python wheel includes:

- Compiled C++ libraries
- All dependencies (self-contained)
- Visualization assets
- ML model weights

This ensures a complete, working installation without external dependencies.

Can I use ACloudViewer with Jupyter?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Yes! ACloudViewer supports Jupyter notebooks:

.. code-block:: python

   import cloudViewer as cv3d
   cv3d.visualization.draw_geometries([pcd])

See the :doc:`/tutorial/index` for Jupyter examples.

How do I integrate with NumPy/PyTorch/TensorFlow?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ACloudViewer provides seamless integration:

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np
   
   # NumPy array to PointCloud
   points = np.random.rand(1000, 3)
   pcd = cv3d.geometry.PointCloud()
   pcd.points = cv3d.utility.Vector3dVector(points)
   
   # PointCloud to NumPy array
   points_np = np.asarray(pcd.points)

See :doc:`/tutorial/geometry/pointcloud` for more examples.

Performance Questions
---------------------

How do I enable GPU acceleration?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

GPU acceleration is automatic when CUDA is available:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Check CUDA availability
   print(cv3d.core.cuda.is_available())
   
   # Use GPU device
   device = cv3d.core.Device("CUDA:0")

Why is processing slow?
^^^^^^^^^^^^^^^^^^^^^^^^

Common reasons:

1. **Large data**: Downsample point clouds first
2. **No GPU**: Enable CUDA for acceleration
3. **Debug build**: Use Release build for production
4. **Single-threaded**: Enable OpenMP parallelization

Optimization tips:

.. code-block:: python

   # Downsample large point clouds
   pcd = pcd.voxel_down_sample(voxel_size=0.01)
   
   # Use GPU
   pcd_gpu = pcd.to(cv3d.core.Device("CUDA:0"))
   
   # Parallel processing
   import cv3d.pipelines.registration as registration
   result = registration.icp(source, target, 
                              num_threads=8)

Troubleshooting
---------------

ImportError: No module named 'cloudViewer'
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Solutions:**

1. Verify installation: ``pip list | grep cloudviewer``
2. Check Python version: ``python --version`` (must be 3.10-3.12)
3. Reinstall: ``pip install --force-reinstall cloudviewer-x.x.x.whl``

Segmentation fault or crash
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Common causes:**

1. **Graphics driver**: Update to latest drivers
2. **Memory**: Large datasets may exceed available RAM
3. **ABI mismatch**: Ensure consistent C++ ABI (``_GLIBCXX_USE_CXX11_ABI``)

**Solutions:**

- Update graphics drivers
- Reduce data size with downsampling
- Rebuild from source if linking errors occur

Visualization window doesn't open
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Linux-specific:**

Ensure X11 or Wayland is running. For headless servers:

.. code-block:: bash

   # Use offscreen rendering
   export DISPLAY=:0
   # Or use EGL
   export PYOPENGL_PLATFORM=egl

**macOS-specific:**

Ensure GUI frameworks are properly installed (part of Xcode).

Where to Get Help
-----------------

- **Documentation**: Browse the complete `documentation </documentation/>`_
- **Issues**: Report bugs on `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_
- **Discussions**: Ask questions on `GitHub Discussions <https://github.com/Asher-1/ACloudViewer/discussions>`_
- **Examples**: Check ``examples/`` directory for code samples
- **Community**: Join our community (see :doc:`support`)

Can't find your question?
^^^^^^^^^^^^^^^^^^^^^^^^^^

If your question isn't answered here:

1. Search `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_
2. Ask on `GitHub Discussions <https://github.com/Asher-1/ACloudViewer/discussions>`_
3. Check the :doc:`support` page for more resources
