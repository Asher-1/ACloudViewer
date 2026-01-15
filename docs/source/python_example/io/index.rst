I/O Examples
============

Python examples for input/output operations.

Point Cloud I/O
---------------

Reading and writing point clouds:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Read point cloud
   pcd = cv3d.io.read_point_cloud("pointcloud.pcd")
   
   # Write point cloud
   cv3d.io.write_point_cloud("output.ply", pcd)

Mesh I/O
--------

Reading and writing triangle meshes:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Read mesh
   mesh = cv3d.io.read_triangle_mesh("mesh.ply")
   
   # Write mesh
   cv3d.io.write_triangle_mesh("output.obj", mesh)

Image I/O
---------

Reading and writing images:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Read image
   image = cv3d.io.read_image("image.png")
   
   # Write image
   cv3d.io.write_image("output.jpg", image)

Camera Parameters I/O
---------------------

Reading and writing camera intrinsics and trajectories:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Read camera intrinsic
   intrinsic = cv3d.io.read_pinhole_camera_intrinsic("camera.json")
   
   # Read camera trajectory
   trajectory = cv3d.io.read_pinhole_camera_trajectory("trajectory.json")
   
   # Write camera parameters
   cv3d.io.write_pinhole_camera_intrinsic("output.json", intrinsic)
   cv3d.io.write_pinhole_camera_trajectory("output.json", trajectory)

File Format Support
-------------------

**Point Clouds:**

- PCD, PLY, XYZ, XYZN, XYZRGB, PTS

**Meshes:**

- PLY, OBJ, STL, OFF, GLTF, GLB

**Images:**

- PNG, JPG, JPEG

**Camera Parameters:**

- JSON format for intrinsics and trajectories

Example Code
------------

For complete runnable examples, see:

**Point Cloud I/O:**
- `point_cloud_io.py <../../../examples/Python/io/point_cloud_io.py>`_

**Triangle Mesh I/O:**
- `triangle_mesh_io.py <../../../examples/Python/io/triangle_mesh_io.py>`_

**Image I/O:**
- `image_io.py <../../../examples/Python/io/image_io.py>`_

**RealSense I/O:**
- `realsense_io.py <../../../examples/Python/io/realsense_io.py>`_

Tutorials
---------

.. toctree::
   :maxdepth: 1
   
   ../../tutorial/geometry/file_io

See Also
--------

- :doc:`../../tutorial/geometry/pointcloud` - Point Cloud Tutorial
- :doc:`../../tutorial/geometry/mesh` - Mesh Tutorial
- :doc:`../../python_api/cloudViewer.io` - I/O API Reference

