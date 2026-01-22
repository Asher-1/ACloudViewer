Point Cloud Registration
=========================

Learn how to align point clouds using various registration algorithms.

ICP Registration
----------------

**Point-to-Point ICP:**

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np
   
   # Load and prepare point clouds
   source = cv3d.io.read_point_cloud("source.pcd")
   target = cv3d.io.read_point_cloud("target.pcd")
   
   # Downsample
   source = source.voxel_down_sample(0.05)
   target = target.voxel_down_sample(0.05)
   
   # Estimate normals
   source.estimate_normals(
       search_param=cv3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
   )
   target.estimate_normals(
       search_param=cv3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
   )
   
   # ICP registration
   threshold = 0.02
   trans_init = np.eye(4)
   
   result = cv3d.pipelines.registration.registration_icp(
       source, target, threshold, trans_init,
       cv3d.pipelines.registration.TransformationEstimationPointToPoint()
   )
   
   print(f"Fitness: {result.fitness}")
   print(f"RMSE: {result.inlier_rmse}")
   print(f"Transformation:\n{result.transformation}")

Interactive Tutorials
---------------------

For detailed, interactive examples, see:

- ICP Registration Tutorial
- Colored Point Cloud ICP
- Global Registration
- Multi-way Registration
- Robust Kernels

These tutorials are available as Jupyter notebooks in the repository.

.. seealso::

   - :doc:`../geometry/pointcloud` - Point Cloud Basics
   - :doc:`../../python_api/cloudViewer.pipelines` - Pipelines API

