Reconstruction System Examples
==============================

Python examples for the complete RGB-D reconstruction system pipeline.

Overview
--------

The reconstruction system provides a complete pipeline for reconstructing 3D scenes from RGB-D sequences:

- **Make Fragments**: Build local geometric models from RGB-D sequences
- **Register Fragments**: Align fragments using global registration
- **Refine Registration**: Optimize fragment alignment
- **Integrate Scene**: Integrate all fragments into a unified scene
- **Color Map Optimization**: Optimize color mapping for the final mesh

Example Code
------------

For complete runnable examples, see:

**Main Pipeline:**
- `run_system.py <../../../examples/Python/reconstruction_system/run_system.py>`_ - Complete reconstruction pipeline
- `make_fragments.py <../../../examples/Python/reconstruction_system/make_fragments.py>`_ - Create fragments from RGB-D sequences
- `register_fragments.py <../../../examples/Python/reconstruction_system/register_fragments.py>`_ - Register fragments globally
- `refine_registration.py <../../../examples/Python/reconstruction_system/refine_registration.py>`_ - Refine fragment registration
- `integrate_scene.py <../../../examples/Python/reconstruction_system/integrate_scene.py>`_ - Integrate all fragments
- `optimize_posegraph.py <../../../examples/Python/reconstruction_system/optimize_posegraph.py>`_ - Optimize pose graph

**Color Optimization:**
- `color_map_optimization_for_reconstruction_system.py <../../../examples/Python/reconstruction_system/color_map_optimization_for_reconstruction_system.py>`_ - Color map optimization

**SLAC (Simultaneous Localization and Calibration):**
- `slac.py <../../../examples/Python/reconstruction_system/slac.py>`_ - SLAC algorithm
- `slac_integrate.py <../../../examples/Python/reconstruction_system/slac_integrate.py>`_ - SLAC integration

**Utilities:**
- `data_loader.py <../../../examples/Python/reconstruction_system/data_loader.py>`_ - Data loading utilities
- `initialize_config.py <../../../examples/Python/reconstruction_system/initialize_config.py>`_ - Configuration initialization
- `opencv_pose_estimation.py <../../../examples/Python/reconstruction_system/opencv_pose_estimation.py>`_ - OpenCV pose estimation

**Sensors:**
- `sensors/azure_kinect_viewer.py <../../../examples/Python/reconstruction_system/sensors/azure_kinect_viewer.py>`_ - Azure Kinect viewer
- `sensors/azure_kinect_recorder.py <../../../examples/Python/reconstruction_system/sensors/azure_kinect_recorder.py>`_ - Azure Kinect recorder
- `sensors/azure_kinect_mkv_reader.py <../../../examples/Python/reconstruction_system/sensors/azure_kinect_mkv_reader.py>`_ - Azure Kinect MKV reader
- `sensors/realsense_recorder.py <../../../examples/Python/reconstruction_system/sensors/realsense_recorder.py>`_ - RealSense recorder
- `sensors/realsense_pcd_visualizer.py <../../../examples/Python/reconstruction_system/sensors/realsense_pcd_visualizer.py>`_ - RealSense point cloud visualizer
- `sensors/realsense_helper.py <../../../examples/Python/reconstruction_system/sensors/realsense_helper.py>`_ - RealSense helper utilities

**Debug Tools:**
- `debug/visualize_fragments.py <../../../examples/Python/reconstruction_system/debug/visualize_fragments.py>`_ - Visualize fragments
- `debug/visualize_scene.py <../../../examples/Python/reconstruction_system/debug/visualize_scene.py>`_ - Visualize scene
- `debug/visualize_pointcloud.py <../../../examples/Python/reconstruction_system/debug/visualize_pointcloud.py>`_ - Visualize point cloud
- `debug/visualize_alignment.py <../../../examples/Python/reconstruction_system/debug/visualize_alignment.py>`_ - Visualize alignment
- `debug/pairwise_pc_alignment.py <../../../examples/Python/reconstruction_system/debug/pairwise_pc_alignment.py>`_ - Pairwise point cloud alignment
- `debug/pairwise_rgbd_alignment.py <../../../examples/Python/reconstruction_system/debug/pairwise_rgbd_alignment.py>`_ - Pairwise RGB-D alignment

Configuration
-------------

Example configuration files:
- `config/tutorial.json <../../../examples/Python/reconstruction_system/config/tutorial.json>`_ - Tutorial configuration
- `config/realsense.json <../../../examples/Python/reconstruction_system/config/realsense.json>`_ - RealSense configuration

See Also
--------

- :doc:`../../tutorial/reconstruction_system/index` - Reconstruction System Tutorials
- :doc:`../../tutorial/sensor/index` - Sensor Integration Tutorials
- :doc:`../../python_api/cloudViewer.reconstruction` - Reconstruction API Reference
