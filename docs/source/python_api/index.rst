.. _python_api:

Python API Documentation
========================

Welcome to the ACloudViewer Python API documentation.

The ACloudViewer Python API provides a comprehensive interface for 3D point cloud and mesh processing. All major features are accessible through Python bindings, making it easy to prototype, experiment, and deploy 3D processing pipelines.

Quick Start
-----------

.. code-block:: python

   import cloudViewer as cv3d
   
   # Load a point cloud
   pcd = cv3d.io.read_point_cloud("cloud.pcd")
   print(f"Loaded {len(pcd.points)} points")
   
   # Visualize
   cv3d.visualization.draw_geometries([pcd])

API Modules
-----------

.. toctree::
   :maxdepth: 2

   cloudViewer.camera
   cloudViewer.core
   cloudViewer.data
   cloudViewer.geometry
   cloudViewer.io
   cloudViewer.ml
   cloudViewer.pipelines
   cloudViewer.reconstruction
   cloudViewer.t
   cloudViewer.utility
   cloudViewer.visualization

Module Overview
---------------

**Core Modules:**

- :doc:`cloudViewer.geometry` - 3D data structures (point clouds, meshes, octrees)
- :doc:`cloudViewer.io` - File I/O operations for various formats
- :doc:`cloudViewer.visualization` - Interactive 3D visualization
- :doc:`cloudViewer.pipelines` - High-level processing pipelines (registration, segmentation)

**Advanced Modules:**

- :doc:`cloudViewer.ml` - Machine learning support with PyTorch/TensorFlow integration
- :doc:`cloudViewer.t` - Tensor-based operations for GPU acceleration
- :doc:`cloudViewer.reconstruction` - 3D reconstruction algorithms
- :doc:`cloudViewer.camera` - Camera models and calibration
- :doc:`cloudViewer.core` - Low-level tensor operations
- :doc:`cloudViewer.utility` - Utility functions and helpers
- :doc:`cloudViewer.data` - Sample datasets for testing

See Also
--------

- :doc:`../tutorial/index` - Step-by-step tutorials
- :doc:`../python_example/geometry/index` - Geometry examples
- :doc:`../python_example/visualization/index` - Visualization examples
- :doc:`../cpp_api` - C++ API documentation
- :doc:`../getting_started/quickstart` - Quick start guide
