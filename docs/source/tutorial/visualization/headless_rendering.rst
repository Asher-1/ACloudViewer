.. _headless_rendering:

Headless rendering
-------------------------------------

Headless rendering allows you to render 3D scenes without a display, which is useful for server environments or automated rendering pipelines.

Offscreen rendering
````````````````````````````````````

ACloudViewer supports offscreen rendering using OSMesa or EGL backends:

.. code-block:: python

    import cloudViewer as cv3d
    
    # Create offscreen visualizer
    vis = cv3d.visualization.Visualizer()
    vis.create_window(visible=False)  # Create offscreen window
    
    # Add geometry
    pcd = cv3d.io.read_point_cloud("cloud.pcd")
    vis.add_geometry(pcd)
    
    # Render to image
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer()
    
    # Save image
    cv3d.io.write_image("render.png", image)
    
    vis.destroy_window()

Rendering to file
````````````````````````````````````

You can render multiple views and save them:

.. code-block:: python

    import cloudViewer as cv3d
    import numpy as np
    
    vis = cv3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    
    # Render from different viewpoints
    for i, angle in enumerate(np.linspace(0, 2*np.pi, 36)):
        vis.get_view_control().rotate(10, 0)
        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer()
        cv3d.io.write_image(f"render_{i:03d}.png", image)
    
    vis.destroy_window()

.. seealso::

   - :doc:`visualization` - Basic visualization
   - :doc:`cpu_rendering` - CPU rendering
   - :doc:`../../python_api/cloudViewer.visualization` - Visualization API
