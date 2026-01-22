.. _cpu_rendering:

CPU rendering
-------------------------------------

ACloudViewer supports CPU-based rendering for systems without GPU support or for specific use cases.

CPU rendering backend
````````````````````````````````````

CPU rendering uses software-based rendering which doesn't require GPU acceleration:

.. code-block:: python

    import cloudViewer as cv3d
    
    # Create visualizer with CPU rendering
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    
    # Set rendering option for CPU rendering
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    
    # Add geometry
    pcd = cv3d.io.read_point_cloud("cloud.pcd")
    vis.add_geometry(pcd)
    
    vis.run()
    vis.destroy_window()

Performance considerations
````````````````````````````````````

CPU rendering is generally slower than GPU rendering but provides:
- Compatibility with systems without GPU
- Deterministic rendering results
- Easier debugging

For large point clouds or meshes, consider:
- Using voxel downsampling
- Reducing point/mesh size
- Using appropriate rendering options

.. seealso::

   - :doc:`visualization` - Basic visualization
   - :doc:`headless_rendering` - Headless rendering
   - :doc:`../../python_api/cloudViewer.visualization` - Visualization API
