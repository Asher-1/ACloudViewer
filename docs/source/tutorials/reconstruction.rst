Reconstruction
==============

This section is under development. For detailed tutorials, see :doc:`../tutorial/reconstruction/index`.

Python API References
---------------------

Key modules for 3D reconstruction:

* :doc:`../python_api/cloudViewer.reconstruction` - Reconstruction algorithms
* :doc:`../python_api/cloudViewer.geometry` - Mesh and surface operations
* :doc:`../python_api/cloudViewer.pipelines` - RGBD and SLAM pipelines

Reconstruction Methods
----------------------

**Surface Reconstruction:**

.. code-block:: python

   import cloudViewer as cv3d
   
   pcd = cv3d.io.read_point_cloud("cloud.pcd")
   pcd.estimate_normals()
   
   # Poisson surface reconstruction
   mesh, densities = cv3d.geometry.ccMesh.create_from_point_cloud_poisson(
       pcd, depth=9)
   
   # Ball pivoting
   radii = [0.005, 0.01, 0.02, 0.04]
   mesh = cv3d.geometry.ccMesh.create_from_point_cloud_ball_pivoting(
       pcd, cv3d.utility.DoubleVector(radii))
   
   # Alpha shapes
   alpha = 0.03
   mesh = cv3d.geometry.ccMesh.create_from_point_cloud_alpha_shape(
       pcd, alpha)

**RGBD Integration:**

.. code-block:: python

   # TSDF Volume integration
   volume = cv3d.pipelines.integration.ScalableTSDFVolume(
       voxel_length=4.0 / 512.0,
       sdf_trunc=0.04,
       color_type=cv3d.pipelines.integration.TSDFVolumeColorType.RGB8)
   
   for i in range(len(rgbd_images)):
       volume.integrate(
           rgbd_images[i],
           cv3d.camera.PinholeCameraIntrinsic(),
           np.linalg.inv(poses[i]))
   
   mesh = volume.extract_triangle_mesh()

See Also
--------

* :doc:`../tutorial/reconstruction/surface_reconstruction` - Surface reconstruction
* :doc:`../tutorial/reconstruction/rgbd_integration` - RGBD integration
* :doc:`../python_api/index` - Complete API reference
