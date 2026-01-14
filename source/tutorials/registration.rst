Registration
============

This section is under development. For detailed tutorials, see :doc:`../tutorial/pipelines/index`.

Python API References
---------------------

Key modules for point cloud registration:

* :doc:`../python_api/cloudViewer.pipelines` - Registration algorithms (ICP, RANSAC, etc.)
* :doc:`../python_api/cloudViewer.t` - Tensor-based registration (GPU-accelerated)
* :doc:`../python_api/cloudViewer.geometry` - Transformation utilities

Registration Methods
--------------------

**ICP (Iterative Closest Point):**

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np
   
   source = cv3d.io.read_point_cloud("source.pcd")
   target = cv3d.io.read_point_cloud("target.pcd")
   
   # Initial alignment
   threshold = 0.02
   trans_init = np.eye(4)
   
   # Point-to-point ICP
   reg_p2p = cv3d.pipelines.registration.registration_icp(
       source, target, threshold, trans_init,
       cv3d.pipelines.registration.TransformationEstimationPointToPoint())
   
   # Point-to-plane ICP
   source.estimate_normals()
   target.estimate_normals()
   reg_p2plane = cv3d.pipelines.registration.registration_icp(
       source, target, threshold, trans_init,
       cv3d.pipelines.registration.TransformationEstimationPointToPlane())

**Global Registration:**

.. code-block:: python

   # RANSAC-based global registration
   result = cv3d.pipelines.registration.registration_ransac_based_on_feature_matching(
       source, target, source_fpfh, target_fpfh,
       mutual_filter=True,
       max_correspondence_distance=1.5,
       estimation_method=cv3d.pipelines.registration.TransformationEstimationPointToPoint(False),
       ransac_n=4,
       criteria=cv3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))

See Also
--------

* :doc:`../tutorial/pipelines/icp_registration` - ICP registration tutorial
* :doc:`../tutorial/pipelines/global_registration` - Global registration
* :doc:`../python_api/index` - Complete API reference
