.. _non_blocking_visualization:

Non-blocking visualization
-------------------------------------

By default, ``draw_geometries`` blocks execution until the window is closed. For applications that need to update the visualization continuously, non-blocking visualization is required.

Using Visualizer class
````````````````````````````````````

The ``Visualizer`` class provides non-blocking visualization capabilities:

.. code-block:: python

    import cloudViewer as cv3d
    import time
    
    # Create visualizer
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometry
    pcd = cv3d.io.read_point_cloud("cloud.pcd")
    vis.add_geometry(pcd)
    
    # Non-blocking update loop
    for i in range(100):
        # Update geometry
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
    
    vis.destroy_window()

Key callbacks
````````````````````````````````````

You can register key callbacks for interactive control:

.. code-block:: python

    import cloudViewer as cv3d
    
    vis = cv3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.add_geometry(pcd)
    
    def key_callback(vis, key, action):
        if key == ord('Q') and action == 0:  # Q key released
            vis.destroy_window()
            return False
        return True
    
    vis.register_key_callback(ord('Q'), key_callback)
    vis.run()

.. seealso::

   - :doc:`visualization` - Basic visualization
   - :doc:`customized_visualization` - Customized visualization
   - :doc:`../../python_api/cloudViewer.visualization` - Visualization API
