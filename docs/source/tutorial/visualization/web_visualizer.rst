.. _web_visualizer:

Web visualizer
-------------------------------------

ACloudViewer supports web-based visualization through WebRTC, allowing you to view 3D scenes in a web browser.

WebRTC visualization
````````````````````````````````````

The web visualizer enables remote visualization and sharing of 3D scenes:

.. code-block:: python

    import cloudViewer as cv3d
    
    # Create web visualizer
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometry
    pcd = cv3d.io.read_point_cloud("cloud.pcd")
    vis.add_geometry(pcd)
    
    # Start web server
    vis.start_webrtc_server(port=8888)
    
    # Access visualization at http://localhost:8888
    vis.run()
    vis.destroy_window()

Features
````````````````````````````````````

The web visualizer provides:
- Real-time 3D rendering in browser
- Interactive controls (rotation, zoom, pan)
- Remote access capabilities
- Cross-platform compatibility

.. seealso::

   - :doc:`visualization` - Basic visualization
   - :doc:`../../python_api/cloudViewer.visualization` - Visualization API
