Sensor
======

This section covers sensor integration and data capture with ACloudViewer. ACloudViewer supports various RGB-D sensors for capturing and processing 3D data.

Supported Sensors
-----------------

.. toctree::
   :maxdepth: 1

   azure_kinect
   realsense

Overview
--------

ACloudViewer provides comprehensive support for RGB-D sensors:

- **Azure Kinect**: Microsoft Azure Kinect DK sensor support
- **RealSense**: Intel RealSense camera support (L515, D435, etc.)

Key Features
------------

- Live RGB-D capture
- Recording to various formats (MKV, bag files, image sequences)
- Camera calibration and intrinsic parameter handling
- Synchronized color and depth streams
- Alignment and preprocessing utilities

Use Cases
---------

- Dataset capture for reconstruction
- Real-time visualization
- SLAM and tracking applications
- 3D scanning workflows

.. seealso::

   - :doc:`../reconstruction_system/capture_your_own_dataset` - Capturing datasets
   - :doc:`../../python_api/cloudViewer.io` - I/O API Reference
   - :doc:`../../python_api/cloudViewer.t.io` - Tensor I/O API Reference
