Point Cloud Processing
======================

This section is under development. For detailed tutorials, see :doc:`../tutorial/geometry/pointcloud`.

Python API References
---------------------

Key modules for point cloud processing:

* :doc:`../python_api/cloudViewer.geometry` - Point cloud data structures
* :doc:`../python_api/cloudViewer.pipelines` - Processing pipelines
* :doc:`../python_api/cloudViewer.utility` - Helper functions

Common Operations
-----------------

**Filtering:**

.. code-block:: python

   import cloudViewer as cv3d
   
   pcd = cv3d.io.read_point_cloud("cloud.pcd")
   
   # Voxel downsampling
   pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
   
   # Statistical outlier removal
   pcd_clean, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
   
   # Radius outlier removal
   pcd_clean, ind = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

**Normal Estimation:**

.. code-block:: python

   # Estimate normals
   pcd.estimate_normals(
       search_param=cv3d.geometry.KDTreeSearchParamHybrid(
           radius=0.1, max_nn=30))

See Also
--------

* :doc:`../tutorial/geometry/pointcloud_outlier_removal` - Outlier removal tutorial
* :doc:`../python_example/geometry/index` - Geometry examples
* :doc:`../python_api/index` - Complete API reference
