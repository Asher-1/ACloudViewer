Camera Examples
================

Python examples for camera intrinsics, extrinsics, and trajectory handling.

Camera Intrinsics
-----------------

Working with pinhole camera intrinsics:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Create default camera intrinsic
   intrinsic = cv3d.camera.PinholeCameraIntrinsic(
       cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
   
   # Create custom camera intrinsic
   intrinsic = cv3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
   
   # Save and load camera intrinsic
   cv3d.io.write_pinhole_camera_intrinsic("camera.json", intrinsic)
   loaded = cv3d.io.read_pinhole_camera_intrinsic("camera.json")

Camera Trajectory
-----------------

Reading and working with camera trajectories:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Read camera trajectory from file
   trajectory = cv3d.io.read_pinhole_camera_trajectory("trajectory.json")
   
   # Access camera parameters
   for param in trajectory.parameters:
       print(f"Intrinsic: {param.intrinsic}")
       print(f"Extrinsic: {param.extrinsic}")
   
   # Save trajectory
   cv3d.io.write_pinhole_camera_trajectory("output.json", trajectory)

RGB-D Reconstruction
--------------------

Using camera parameters for RGB-D reconstruction:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Load RGB-D dataset
   dataset = cv3d.data.SampleRedwoodRGBDImages()
   trajectory = cv3d.io.read_pinhole_camera_trajectory(
       dataset.trajectory_log_path)
   
   # Create point clouds from RGB-D images
   pcds = []
   for i in range(len(dataset.depth_paths)):
       depth = cv3d.io.read_image(dataset.depth_paths[i])
       color = cv3d.io.read_image(dataset.color_paths[i])
       rgbd = cv3d.geometry.RGBDImage.create_from_color_and_depth(
           color, depth, 1000.0, 5.0, False)
       pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
           rgbd, 
           trajectory.parameters[i].intrinsic,
           trajectory.parameters[i].extrinsic)
       pcds.append(pcd)
   
   cv3d.visualization.draw_geometries(pcds)

Example Code
------------

For a complete example, see:
- `examples/Python/camera/camera_trajectory.py <../../../examples/Python/camera/camera_trajectory.py>`_

See Also
--------

- :doc:`../../tutorial/geometry/rgbd_image` - RGB-D Image Tutorial
- :doc:`../../tutorial/pipelines/rgbd_integration` - RGB-D Integration
- :doc:`../../python_api/cloudViewer.camera` - Camera API Reference
- :doc:`../../python_api/cloudViewer.io` - I/O API Reference
