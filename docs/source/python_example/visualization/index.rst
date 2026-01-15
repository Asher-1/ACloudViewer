Visualization Examples
======================

Python examples for 3D visualization and rendering.

Basic Visualization
--------------------

Simple visualization examples:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Load and visualize point cloud
   pcd = cv3d.io.read_point_cloud("pointcloud.pcd")
   cv3d.visualization.draw_geometries([pcd])
   
   # Visualize multiple geometries
   mesh = cv3d.io.read_triangle_mesh("mesh.ply")
   cv3d.visualization.draw_geometries([pcd, mesh])

Interactive Visualization
-------------------------

Advanced visualization with custom controls:

.. code-block:: python

   import cloudViewer as cv3d
   
   # Create visualizer
   vis = cv3d.visualization.Visualizer()
   vis.create_window()
   vis.add_geometry(pcd)
   
   # Custom rendering options
   opt = vis.get_render_option()
   opt.point_size = 2.0
   opt.background_color = [1.0, 1.0, 1.0]
   
   vis.run()
   vis.destroy_window()

Tutorials
---------

.. toctree::
   :maxdepth: 1
   
   ../../tutorial/visualization/visualization

Example Code
------------

For complete runnable examples, see:

**Basic Visualization:**
- `visualization.py <../../../examples/Python/visualization/visualization.py>`_
- `draw.py <../../../examples/Python/visualization/draw.py>`_
- `draw_webrtc.py <../../../examples/Python/visualization/draw_webrtc.py>`_

**Interactive Visualization:**
- `interactive_visualization.py <../../../examples/Python/visualization/interactive_visualization.py>`_
- `customized_visualization.py <../../../examples/Python/visualization/customized_visualization.py>`_
- `customized_visualization_key_action.py <../../../examples/Python/visualization/customized_visualization_key_action.py>`_
- `non_blocking_visualization.py <../../../examples/Python/visualization/non_blocking_visualization.py>`_

**Advanced Visualization:**
- `headless_rendering.py <../../../examples/Python/visualization/headless_rendering.py>`_
- `headless_rendering_in_filament.py <../../../examples/Python/visualization/headless_rendering_in_filament.py>`_
- `render-to-image.py <../../../examples/Python/visualization/render-to-image.py>`_
- `textured_mesh.py <../../../examples/Python/visualization/textured_mesh.py>`_
- `textured-model.py <../../../examples/Python/visualization/textured-model.py>`_
- `text3d.py <../../../examples/Python/visualization/text3d.py>`_
- `mitsuba_material_estimation.py <../../../examples/Python/visualization/mitsuba_material_estimation.py>`_
- `to_mitsuba.py <../../../examples/Python/visualization/to_mitsuba.py>`_
- `video.py <../../../examples/Python/visualization/video.py>`_

**Visualization Tools:**
- `add-geometry.py <../../../examples/Python/visualization/add-geometry.py>`_
- `remove_geometry.py <../../../examples/Python/visualization/remove_geometry.py>`_
- `load_save_viewpoint.py <../../../examples/Python/visualization/load_save_viewpoint.py>`_
- `multiple-windows.py <../../../examples/Python/visualization/multiple-windows.py>`_
- `vis-gui.py <../../../examples/Python/visualization/vis-gui.py>`_
- `all-widgets.py <../../../examples/Python/visualization/all-widgets.py>`_
- `demo_scene.py <../../../examples/Python/visualization/demo_scene.py>`_
- `line-width.py <../../../examples/Python/visualization/line-width.py>`_
- `mouse-and-point-coord.py <../../../examples/Python/visualization/mouse-and-point-coord.py>`_
- `non-english.py <../../../examples/Python/visualization/non-english.py>`_
- `online_processing.py <../../../examples/Python/visualization/online_processing.py>`_
- `remote_visualizer.py <../../../examples/Python/visualization/remote_visualizer.py>`_

**TensorBoard Integration:**
- `tensorboard_pytorch.py <../../../examples/Python/visualization/tensorboard_pytorch.py>`_
- `tensorboard_tensorflow.py <../../../examples/Python/visualization/tensorboard_tensorflow.py>`_

See Also
--------

- :doc:`../../tutorial/visualization/index` - Visualization Tutorials
- :doc:`../../python_api/cloudViewer.visualization` - Visualization API Reference

