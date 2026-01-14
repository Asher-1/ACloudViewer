Visualization
=============


This section is under development. For detailed tutorials, see :doc:`../tutorial/visualization/index`.

Python API References
---------------------

Key modules for visualization:

* :doc:`../python_api/cloudViewer.visualization` - 3D visualization and rendering
* :doc:`../python_api/cloudViewer.geometry` - Geometry for visualization
* :doc:`../python_api/cloudViewer.utility` - Color maps and utilities

Visualization Methods
---------------------

**Basic Visualization:**

.. code-block:: python

   import cloudViewer as cv3d
   
   # Simple visualization
   pcd = cv3d.io.read_point_cloud("cloud.pcd")
   cv3d.visualization.draw_geometries([pcd])
   
   # Multiple geometries
   mesh = cv3d.io.read_triangle_mesh("mesh.ply")
   cv3d.visualization.draw_geometries([pcd, mesh])
   
   # With custom window name
   cv3d.visualization.draw_geometries(
       [pcd],
       window_name="Point Cloud",
       width=1920,
       height=1080)

**Advanced Visualization:**

.. code-block:: python

   # Non-blocking visualization
   vis = cv3d.visualization.Visualizer()
   vis.create_window()
   vis.add_geometry(pcd)
   vis.run()
   vis.destroy_window()
   
   # Custom view control
   vis = cv3d.visualization.Visualizer()
   vis.create_window()
   vis.add_geometry(pcd)
   
   # Get view control
   ctr = vis.get_view_control()
   ctr.set_zoom(0.8)
   ctr.set_lookat([0, 0, 0])
   ctr.set_up([0, 0, 1])
   
   vis.run()
   vis.destroy_window()

**Jupyter Visualization:**

.. code-block:: python

   # In Jupyter notebooks
   from cloudViewer.web_visualizer import draw
   
   pcd = cv3d.io.read_point_cloud("cloud.pcd")
   draw(pcd)

See Also
--------

* :doc:`../tutorial/visualization/visualization` - Basic visualization
* :doc:`../tutorial/visualization/jupyter_visualization` - Jupyter integration
* :doc:`../python_example/visualization/index` - Visualization examples
* :doc:`../python_api/index` - Complete API reference
