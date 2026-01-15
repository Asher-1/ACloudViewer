.. _t_reconstruction_system:

Reconstruction system (Tensor)
===================================================================

This tutorial demonstrates volumetric RGB-D reconstruction and dense RGB-D SLAM with the ACloudViewer tensor interface and hash map backend.

Overview
--------

The tensor-based reconstruction system provides:

- **GPU Acceleration**: Fast volumetric integration on GPU
- **Hash Map Backend**: Efficient memory usage for large scenes
- **Real-Time SLAM**: Online reconstruction and tracking
- **Scalable**: Handle large-scale indoor and outdoor scenes

Key Components
-------------

- **Voxel Block Grid**: Efficient sparse volumetric representation
- **Integration**: TSDF-based RGB-D integration
- **Ray Casting**: Fast ray casting for visualization and tracking
- **Dense SLAM**: Real-time dense SLAM pipeline

Getting started
---------------

.. note::
   The detailed tutorials for tensor-based reconstruction system are currently under development.
   The following topics will be covered:
   
   - Voxel Block Grid
   - Integration
   - Customized Integration
   - Ray Casting
   - Dense SLAM

   For now, please refer to the legacy reconstruction system tutorials in :doc:`../reconstruction_system/index`.

Basic Usage
-----------

Here's a basic example of using the tensor-based reconstruction system:

.. code-block:: python

    import cloudViewer as cv3d
    
    # Initialize reconstruction system
    # (Implementation details will be added as development progresses)
    
    # For now, use the legacy reconstruction system
    # See ../reconstruction_system/index for details

.. seealso::

   - :doc:`../reconstruction_system/index` - Legacy reconstruction system
   - :doc:`../t_pipelines/index` - Tensor-based pipelines
   - :doc:`../sensor/index` - Sensor integration
   - :doc:`../../python_api/cloudViewer.t.reconstruction` - Tensor reconstruction API
