Structure from Motion
=====================

Overview
--------

Structure from Motion (SfM) reconstructs 3D structure from 2D image sequences.

Workflow
--------

1. **Feature Detection**: Detect keypoints in images
2. **Feature Matching**: Match keypoints across images
3. **Camera Pose Estimation**: Estimate camera positions
4. **Triangulation**: Reconstruct 3D points
5. **Bundle Adjustment**: Refine reconstruction


Detailed SfM tutorial with code examples.

Related Topics
--------------

- :doc:`mvs` - Multi-View Stereo
- :doc:`tsdf` - TSDF Volume Integration
- :doc:`../../jupyter/pipelines/rgbd_integration` - RGBD Integration

See Also
--------

- :doc:`../pipelines/registration` - Point Cloud Registration
- :doc:`../geometry/pointcloud` - Point Cloud Processing

