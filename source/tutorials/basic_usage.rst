Basic Usage
===========

This tutorial is under development. Please refer to :doc:`../getting_started/quickstart` for basic examples.

Python API References
---------------------

Core modules for basic usage:

* :doc:`../python_api/cloudViewer.io` - File I/O operations
* :doc:`../python_api/cloudViewer.geometry` - Point clouds and meshes
* :doc:`../python_api/cloudViewer.visualization` - 3D visualization
* :doc:`../python_api/cloudViewer.utility` - Utility functions

Quick Example
-------------

.. code-block:: python

   import cloudViewer as cv3d
   
   # Load a point cloud
   pcd = cv3d.io.read_point_cloud("cloud.pcd")
   print(f"Loaded {len(pcd.points)} points")
   
   # Visualize
   cv3d.visualization.draw_geometries([pcd])

See Also
--------

* `GitHub Examples <https://github.com/Asher-1/ACloudViewer/tree/main/examples>`_
* :doc:`../python_api/index` - Complete Python API Reference
* :doc:`../tutorial/geometry/pointcloud` - Detailed point cloud tutorial
