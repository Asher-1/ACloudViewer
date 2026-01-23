.. _ray_casting_voxel_block_grid:

Ray Casting in a Voxel Block Grid
---------------------------------

.. note::
   This is NOT ray casting for triangle meshes. Please refer to :doc:`/python_api/cloudViewer.t.geometry.RayCastingScene` for that use case.

Ray casting can be performed in a voxel block grid to generate depth and color images at specific view points without extracting the entire surface. It is useful for frame-to-model tracking, and for differentiable volume rendering.

We provide optimized conventional rendering, and basic support for customized rendering that may be used in differentiable rendering. An example can be found at ``examples/Python/t_reconstruction_system/ray_casting.py``.

Conventional rendering
``````````````````````
From a reconstructed voxel block grid :code:`vbg` from :ref:`optimized_integration`, we can efficiently render the scene given the input depth as a rough range estimate.

.. literalinclude:: ../../../../examples/Python/t_reconstruction_system/ray_casting.py
   :language: python
   :lineno-start: 68
   :lines: 8,69-82

The results could be directly obtained and visualized by

.. literalinclude:: ../../../../examples/Python/t_reconstruction_system/ray_casting.py
   :language: python
   :lineno-start: 83
   :lines: 8,84,86-88,98-105

Customized rendering
`````````````````````
In customized rendering, we manually perform trilinear-interpolation by accessing properties at 8 nearest neighbor voxels with respect to the found surface point per pixel:

.. literalinclude:: ../../../../examples/Python/t_reconstruction_system/ray_casting.py
   :language: python
   :lineno-start: 90
   :lines: 8,91-96,107-108

Since the output is rendered via indices, the rendering process could be rewritten in differentiable engines like PyTorch seamlessly via :doc:`/tutorial/core/tensor`.

