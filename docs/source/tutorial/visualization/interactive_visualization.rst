.. _interactive_visualization:

Interactive visualization
-------------------------------------

This tutorial introduces user interaction features of the visualizer window provided by:-

#. ``cloudViewer.visualization.draw_geometries_with_editing``
#. ``cloudViewer.visualization.VisualizerWithEditing``

.. _crop_geometry:

Crop geometry
````````````````````````````````````

This function simply reads a point cloud and calls ``draw_geometries_with_editing``. This function provides vertex selection and cropping.

.. Note:: ACloudViewer has a ``VisualizerWithEditing`` class that inherits ``Visualizer`` class. It adds graphic user interaction features. Likewise examples in :ref:`customized_visualization`, ``VisualizerWithEditing()`` can be explicitly used instead of ``draw_geometries_with_editing([pcd])``.

Once a geometry is displayed, press ``Y`` twice to align geometry with negative direction of y-axis. After adjusting viewing orientation, press ``K`` to lock screen and to switch to the selection mode.

.. Tip:: The practical step for selecting area is to align the geometry with arbitrary axis using orthographic projection model. This trick makes selection easier, because it avoids self-occlusion hassle due to perspective projection.

To select a region, use either ``mouse drag`` (rectangle selection) or ``ctrl + left mouse click`` (polygon selection).

Note that the selected area is dark shaded. To keep the selected area and discard the rest, press ``C``. A dialog box appears, which can be used to save the cropped geometry.

To finish selection mode, press ``F`` to switch to freeview mode.

.. _manual_registration:

Manual registration
````````````````````````````````````

The visualizer also supports manual registration of point clouds. You can select corresponding points in two point clouds and the system will estimate a transformation between them.

Example code:

.. code-block:: python

    import cloudViewer as cv3d
    
    # Load two point clouds
    pcd1 = cv3d.io.read_point_cloud("cloud1.pcd")
    pcd2 = cv3d.io.read_point_cloud("cloud2.pcd")
    
    # Use interactive editing for manual registration
    cv3d.visualization.draw_geometries_with_editing([pcd1, pcd2])

.. seealso::

   - :doc:`visualization` - Basic visualization
   - :doc:`customized_visualization` - Customized visualization
   - :doc:`../../python_api/cloudViewer.visualization` - Visualization API
