Geometry (Tensor)
=================

This section covers tensor-based geometry operations in ACloudViewer. The tensor interface provides GPU-accelerated operations and efficient memory management for large-scale 3D data processing.

Basics
------

.. toctree::
   :maxdepth: 1

   pointcloud

Overview
--------

The tensor-based geometry module provides:

- **GPU Acceleration**: Leverage CUDA for fast computation
- **Memory Efficiency**: Optimized data structures for large point clouds
- **NumPy Integration**: Seamless conversion between tensor and NumPy arrays
- **Batch Processing**: Process multiple geometries simultaneously

Key Features
------------

- Point cloud operations (downsampling, filtering, transformation)
- Mesh operations (creation, manipulation, processing)
- Efficient nearest neighbor search
- Ray casting and distance queries

.. seealso::

   - :doc:`../geometry/index` - Legacy geometry operations
   - :doc:`../../python_api/cloudViewer.t.geometry` - Tensor geometry API
   - :doc:`../t_pipelines/index` - Tensor-based pipelines
