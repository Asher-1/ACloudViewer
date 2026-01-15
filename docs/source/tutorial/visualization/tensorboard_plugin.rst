.. _tensorboard_plugin:

TensorBoard plugin
-------------------------------------

ACloudViewer provides a TensorBoard plugin for visualizing 3D geometries within TensorBoard.

TensorBoard integration
````````````````````````````````````

You can log 3D geometries to TensorBoard for visualization:

.. code-block:: python

    import cloudViewer as cv3d
    from torch.utils.tensorboard import SummaryWriter
    
    # Create TensorBoard writer
    writer = SummaryWriter('runs/experiment')
    
    # Load point cloud
    pcd = cv3d.io.read_point_cloud("cloud.pcd")
    
    # Log to TensorBoard
    writer.add_geometry('pointcloud', pcd, global_step=0)
    
    writer.close()

Viewing in TensorBoard
````````````````````````````````````

After logging, you can view the geometries in TensorBoard:

.. code-block:: sh

    tensorboard --logdir runs/experiment

The 3D geometries will appear in the TensorBoard interface with interactive controls.

.. seealso::

   - :doc:`visualization` - Basic visualization
   - :doc:`../../python_api/cloudViewer.visualization` - Visualization API
