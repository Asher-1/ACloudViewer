Pipelines Examples
==================

Python examples for processing pipelines including registration, RGB-D processing, and optimization.

Registration Examples
---------------------

Point cloud registration and alignment:

.. toctree::
   :maxdepth: 1
   
   ../../tutorial/pipelines/icp_registration
   ../../tutorial/pipelines/colored_pointcloud_registration
   ../../tutorial/pipelines/global_registration
   ../../tutorial/pipelines/multiway_registration
   ../../tutorial/pipelines/robust_kernels

RGBD Examples
-------------

RGB-D image processing and integration:

.. toctree::
   :maxdepth: 1
   
   ../../tutorial/pipelines/rgbd_integration
   ../../tutorial/pipelines/rgbd_odometry
   ../../tutorial/pipelines/color_map_optimization

Example Code
------------

For complete runnable examples, see:

**Registration:**
- `icp_registration.py <../../../examples/Python/pipelines/icp_registration.py>`_
- `colored_icp_registration.py <../../../examples/Python/pipelines/colored_icp_registration.py>`_
- `registration_ransac.py <../../../examples/Python/pipelines/registration_ransac.py>`_
- `registration_fgr.py <../../../examples/Python/pipelines/registration_fgr.py>`_
- `multiway_registration.py <../../../examples/Python/pipelines/multiway_registration.py>`_
- `robust_icp.py <../../../examples/Python/pipelines/robust_icp.py>`_
- `doppler_icp_registration.py <../../../examples/Python/pipelines/doppler_icp_registration.py>`_

**RGB-D Processing:**
- `rgbd_odometry.py <../../../examples/Python/pipelines/rgbd_odometry.py>`_
- `rgbd_integration_uniform.py <../../../examples/Python/pipelines/rgbd_integration_uniform.py>`_

**Optimization:**
- `pose_graph_optimization.py <../../../examples/Python/pipelines/pose_graph_optimization.py>`_

See Also
--------

- :doc:`../../tutorial/pipelines/index` - Pipelines Tutorials
- :doc:`../../python_api/cloudViewer.pipelines` - Pipelines API Reference

