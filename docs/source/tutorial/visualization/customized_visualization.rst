.. _customized_visualization:

Customized visualization
-------------------------------------

The usage of ACloudViewer convenient visualization functions ``draw_geometries`` and ``draw_geometries_with_custom_animation`` is straightforward. Everything can be done with the GUI. Press :kbd:`h` inside the visualizer window to see helper information. For more details, see :any:`/tutorial/visualization/visualization.ipynb`.

This tutorial focuses on more advanced functionalities to customize the behavior of the visualizer window. Please refer to examples/Python/visualization/customized_visualization.py to try the following examples.

Mimic draw_geometries() with Visualizer class
````````````````````````````````````````````````````

Class ``Visualizer`` has a couple of variables such as a ``ViewControl`` and a ``RenderOption``. The following function reads a predefined ``RenderOption`` stored in a json file.

Change field of view
````````````````````````````````````
To change field of view of the camera, it is first necessary to get an instance of the visualizer control. To modify the field of view, use ``change_field_of_view``.

The field of view (FoV) can be set to a degree in the range [5,90]. Note that ``change_field_of_view`` adds the specified FoV to the current FoV. By default, the visualizer has an FoV of 60 degrees.

Custom rendering options
````````````````````````````````````

You can customize various rendering options such as:
- Point size
- Line width
- Background color
- Lighting
- Material properties

Example code:

.. code-block:: python

    import cloudViewer as cv3d
    
    # Create visualizer
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Get render option
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = [1.0, 1.0, 1.0]
    
    vis.run()
    vis.destroy_window()

.. seealso::

   - :doc:`visualization` - Basic visualization
   - :doc:`interactive_visualization` - Interactive visualization
   - :doc:`../../python_api/cloudViewer.visualization` - Visualization API
