Introduction
============

Welcome to ACloudViewer
-----------------------

ACloudViewer is an open-source 3D point cloud and triangular mesh processing software library. It supports rapid development of software for processing 3D data.

Overview
--------

ACloudViewer is built on top of several well-established libraries:

* **CloudCompare**: For professional point cloud processing
* **Open3D**: For modern 3D data processing algorithms
* **ParaView/VTK**: For advanced visualization
* **COLMAP**: For structure-from-motion reconstruction
* **PCL**: For additional point cloud processing capabilities

Key Features
------------

Data Processing
~~~~~~~~~~~~~~~

* Point cloud filtering and segmentation
* Surface reconstruction (Poisson, Ball Pivoting, Alpha Shapes)
* Mesh simplification and smoothing
* Normal estimation and feature extraction
* Voxelization and octree operations

Registration
~~~~~~~~~~~~

* ICP (Iterative Closest Point) variants
* Colored point cloud registration
* Global registration (RANSAC, Fast Global Registration)
* Multiway registration

Reconstruction
~~~~~~~~~~~~~~

* COLMAP-based photo reconstruction
* TSDF (Truncated Signed Distance Function) integration
* RGBD reconstruction
* Real-time reconstruction

Visualization
~~~~~~~~~~~~~

* Interactive 3D visualization
* Multiple rendering backends (VTK, OpenGL)
* Headless rendering support
* PBR (Physically Based Rendering) materials

Machine Learning
~~~~~~~~~~~~~~~~

* Integration with PyTorch and TensorFlow
* 3D semantic segmentation
* Point cloud classification
* Custom ML pipelines

System Requirements
-------------------

Minimum Requirements
~~~~~~~~~~~~~~~~~~~~

* **OS**: Ubuntu 18.04+, macOS 10.15+, Windows 10+
* **RAM**: 8 GB
* **CPU**: Intel i5 or equivalent
* **GPU**: Optional (recommended for large datasets)

Recommended Requirements
~~~~~~~~~~~~~~~~~~~~~~~~

* **OS**: Ubuntu 22.04, macOS 12+, Windows 11
* **RAM**: 16 GB or more
* **CPU**: Intel i7/i9 or AMD Ryzen 7/9
* **GPU**: NVIDIA GPU with CUDA support (8GB+ VRAM)

Supported File Formats
----------------------

Input Formats
~~~~~~~~~~~~~

* Point Cloud: PLY, PCD, PTS, XYZ, LAS, LAZ, E57
* Mesh: OBJ, STL, PLY, OFF, FBX, GLTF/GLB
* Image: PNG, JPG, BMP, TIFF
* RGBD: TUM, Redwood, SUN3D formats

Output Formats
~~~~~~~~~~~~~~

* Point Cloud: PLY, PCD, XYZ, PTS
* Mesh: OBJ, STL, PLY, OFF, GLTF/GLB
* Image: PNG, JPG, BMP, TIFF

Next Steps
----------

* :doc:`installation` - Install ACloudViewer
* :doc:`quickstart` - Quick start guide
* :doc:`build_from_source` - Build from source

