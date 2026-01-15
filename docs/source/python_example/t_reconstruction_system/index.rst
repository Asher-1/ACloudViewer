Tensor Reconstruction System Examples
========================================

Python examples for tensor-based volumetric RGB-D reconstruction and dense RGB-D SLAM.

Overview
--------

The tensor-based reconstruction system provides GPU-accelerated reconstruction with:

- **Voxel Block Grid**: Efficient sparse volumetric representation
- **TSDF Integration**: Truncated Signed Distance Function integration
- **Dense SLAM**: Real-time dense simultaneous localization and mapping
- **Ray Casting**: Fast ray casting for visualization and tracking
- **Pose Graph Optimization**: Optimize camera poses

Example Code
------------

For complete runnable examples, see:

**Main Pipeline:**
- `run_system.py <../../../examples/Python/t_reconstruction_system/run_system.py>`_ - Complete tensor reconstruction pipeline
- `integrate.py <../../../examples/Python/t_reconstruction_system/integrate.py>`_ - TSDF integration
- `integrate_custom.py <../../../examples/Python/t_reconstruction_system/integrate_custom.py>`_ - Custom integration with additional properties

**Dense SLAM:**
- `dense_slam.py <../../../examples/Python/t_reconstruction_system/dense_slam.py>`_ - Dense SLAM (command line)
- `dense_slam_gui.py <../../../examples/Python/t_reconstruction_system/dense_slam_gui.py>`_ - Dense SLAM with GUI

**Ray Casting:**
- `ray_casting.py <../../../examples/Python/t_reconstruction_system/ray_casting.py>`_ - Ray casting for visualization

**Odometry:**
- `rgbd_odometry.py <../../../examples/Python/t_reconstruction_system/rgbd_odometry.py>`_ - RGB-D odometry

**Optimization:**
- `pose_graph_optim.py <../../../examples/Python/t_reconstruction_system/pose_graph_optim.py>`_ - Pose graph optimization

**Utilities:**
- `common.py <../../../examples/Python/t_reconstruction_system/common.py>`_ - Common utilities
- `config.py <../../../examples/Python/t_reconstruction_system/config.py>`_ - Configuration handling
- `split_fragments.py <../../../examples/Python/t_reconstruction_system/split_fragments.py>`_ - Fragment splitting

Configuration
-------------

Example configuration files:
- `default_config.yml <../../../examples/Python/t_reconstruction_system/default_config.yml>`_ - Default configuration
- `default_intrinsics.json <../../../examples/Python/t_reconstruction_system/default_intrinsics.json>`_ - Default camera intrinsics

See Also
--------

- :doc:`../../tutorial/t_reconstruction_system/index` - Tensor Reconstruction System Tutorials
- :doc:`../../tutorial/t_pipelines/index` - Tensor Pipelines Tutorials
- :doc:`../../python_api/cloudViewer.t.pipelines` - Tensor Pipelines API Reference
