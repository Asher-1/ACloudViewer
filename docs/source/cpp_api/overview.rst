Overview
========

ACloudViewer is a comprehensive C++ library for 3D point cloud and mesh processing. This page provides an overview of the library architecture and key modules.

Core Features
-------------

ACloudViewer provides the following core features:

- **3D Data Structures**: Point clouds, meshes, octrees, and hierarchical object models
- **3D Data Processing**: Filtering, segmentation, registration, surface reconstruction
- **Scene Reconstruction**: Multi-view stereo, structure from motion
- **Interactive Visualization**: Qt-based 3D viewer with rich interaction
- **Comprehensive I/O**: Support for PCD, PLY, LAS, E57, OBJ, STL, and many other formats
- **Plugin System**: Extensible architecture for custom algorithms
- **PCL Integration**: Seamless interoperability with Point Cloud Library
- **Cross-Platform**: Linux, macOS, and Windows support

Architecture
------------

ACloudViewer is organized into several key modules:

Core Modules
~~~~~~~~~~~~

cloudViewer
^^^^^^^^^^^

Core point cloud and mesh data structures:

- ``ccPointCloud`` - Main point cloud class with support for colors, normals, scalar fields
- ``ccMesh`` - Triangle mesh representation with material support
- ``ccHObject`` - Hierarchical object model for complex scenes
- ``ccGenericPointCloud`` - Abstract point cloud interface
- ``ccOctree`` - Efficient spatial indexing structure

**Key Features**:

- Efficient memory management for large point clouds
- Support for multiple scalar fields per point
- Hierarchical scene graph
- Bounding box and spatial queries

CV_db
^^^^^^

Database and entity management for complex 3D scenes:

- Scene graph management
- Entity serialization/deserialization
- Property management
- Object selection and filtering

**Use Cases**:

- Managing complex multi-object scenes
- Saving and loading project files
- Undo/redo functionality

CV_io
^^^^^^

I/O operations for various file formats:

**Point Cloud Formats**:

- PCD (Point Cloud Data)
- PLY (Polygon File Format)
- LAS/LAZ (LiDAR formats)
- E57 (ASTM standard)
- XYZ, ASCII formats

**Mesh Formats**:

- OBJ (Wavefront)
- STL (Stereolithography)
- OFF (Object File Format)
- FBX, COLLADA, etc.

**Key Classes**:

- ``FileIOFilter`` - Base I/O filter class
- Plugin-based architecture for format extensions

PCLEngine
^^^^^^^^^

Integration with Point Cloud Library (PCL):

**Algorithms**:

- **Filtering**: Statistical outlier removal, voxel grid, pass-through
- **Registration**: ICP, NDT, GICP
- **Segmentation**: RANSAC, region growing, Euclidean clustering
- **Surface Reconstruction**: Poisson, Greedy Projection, MLS

**Key Features**:

- Seamless conversion between ACloudViewer and PCL data types
- Access to 100+ PCL algorithms
- Optimized for large-scale point clouds

Reconstruction
^^^^^^^^^^^^^^

3D reconstruction algorithms:

- **Surface Reconstruction**: Poisson, Ball-Pivoting, Alpha Shapes
- **Mesh Generation**: From point clouds with normals
- **Mesh Processing**: Simplification, smoothing, hole filling

Plugin System
~~~~~~~~~~~~~

CVPluginAPI
^^^^^^^^^^^

Plugin interface definitions:

- ``ccStdPluginInterface`` - Standard plugin interface
- ``ccIOPluginInterface`` - I/O plugin interface for custom formats
- ``ccPclPluginInterface`` - PCL algorithm plugin interface
- ``ccGLFilterPluginInterface`` - OpenGL rendering plugin interface

**Benefits**:

- Extend functionality without modifying core library
- Third-party algorithm integration
- Custom file format support

Standard Plugins
^^^^^^^^^^^^^^^^

ACloudViewer includes many standard plugins:

- **QRANSAC_SD**: RANSAC shape detection
- **QM3C2**: M3C2 distance computation
- **QPoisson_Recon**: Poisson surface reconstruction
- **QCSF**: Cloth Simulation Filtering for ground extraction
- **QAnimation**: Animation and video export
- And 20+ more...

Application Layer
~~~~~~~~~~~~~~~~~

CVViewer
^^^^^^^^

Standalone 3D viewer application:

- Lightweight viewer for quick visualization
- Command-line interface
- Batch processing support

ACloudViewer
^^^

Main ACloudViewer application with full GUI:

- Comprehensive point cloud processing tools
- Interactive 3D visualization
- Plugin management
- Project management (save/load scenes)
- Measurement and annotation tools
- Batch processing

Design Principles
-----------------

Modularity
~~~~~~~~~~

Each module is designed to be as independent as possible, allowing you to use only what you need.

Performance
~~~~~~~~~~~

- Optimized for large-scale data (millions to billions of points)
- Multi-threading where applicable
- Memory-efficient data structures
- GPU acceleration for rendering

Extensibility
~~~~~~~~~~~~~

- Plugin architecture for custom algorithms
- Virtual base classes for easy derivation
- Template-based design for flexibility

Interoperability
~~~~~~~~~~~~~~~~

- Standard file formats
- Integration with PCL
- Python bindings
- C API for maximum compatibility

Getting Started
---------------

To start using ACloudViewer C++:

1. **Review the** :doc:`quickstart` **guide** for basic usage examples
2. **Browse the** `Full API Reference <api/index.html>`_ for detailed class documentation
3. **Check the examples** in ``examples/Cpp/`` directory
4. **Compile from source**: :ref:`compilation`

Next Steps
----------

- :doc:`quickstart` - Quick start guide with code examples
- :doc:`plugins` - Plugin system documentation
- `API Reference <api/index.html>`_ - Complete API documentation
- :ref:`compilation` - Build from source
- :doc:`../tutorial/index` - Python and C++ tutorials

See Also
--------

- `GitHub Repository <https://github.com/Asher-1/ACloudViewer>`_
- :doc:`../tutorial/index` - Tutorials
- :doc:`../examples/cpp_examples` - C++ Examples
