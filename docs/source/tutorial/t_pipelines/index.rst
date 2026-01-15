.. _t_pipelines:

Pipelines (Tensor)
==================

This section covers tensor-based processing pipelines in ACloudViewer. The tensor interface provides GPU-accelerated pipelines for registration, SLAM, and reconstruction.

Registration
------------

.. toctree::
   :maxdepth: 1

   t_icp_registration
   t_robust_kernel

Overview
--------

The tensor-based pipelines module provides:

- **GPU Acceleration**: Fast registration and optimization on GPU
- **Robust Kernels**: Handle outliers and noise in registration
- **Large-Scale Processing**: Efficient processing of large point clouds
- **Real-Time Performance**: Suitable for SLAM and online reconstruction

Key Features
------------

- ICP registration with tensor backend
- Robust kernel functions (Huber, Cauchy, etc.)
- Multi-scale registration
- Pose graph optimization

Related Topics
--------------

- :doc:`../reconstruction_system/index` - Legacy reconstruction system
- :doc:`../t_reconstruction_system/index` - Tensor-based reconstruction system
- :doc:`../sensor/index` - Sensor integration

.. seealso::

   - :doc:`../pipelines/index` - Legacy pipelines
   - :doc:`../../python_api/cloudViewer.t.pipelines` - Tensor pipelines API
