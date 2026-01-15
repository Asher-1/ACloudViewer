.. _azure_kinect:

Azure Kinect with ACloudViewer
-------------------------------

Azure Kinect is supported on Windows and Linux (Ubuntu 18.04+).

Installation
============

Install the Azure Kinect SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Follow `this guide <https://github.com/microsoft/Azure-Kinect-Sensor-SDK>`_
to install the Azure Kinect SDK (K4A).

On Ubuntu, you'll need to set up a udev rule to use the Kinect camera without
``sudo``. Follow
`these instructions <https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/usage.md#linux-device-setup>`_.

After installation, you may run ``k4aviewer`` from the Linux terminal or
``k4aviewer.exe`` on Windows to make sure that the device is working.

Currently, ACloudViewer supports the Azure Kinect SDK version ``v1.4.1``, though future
versions might also be compatible.

Compile from Source
~~~~~~~~~~~~~~~~~~~

To build ACloudViewer from source with K4A support, set ``BUILD_AZURE_KINECT=ON`` at
CMake config step. That is,

.. code-block:: sh

    cmake -DBUILD_AZURE_KINECT=ON -DOTHER_FLAGS ..

ACloudViewer Azure Kinect Viewer
=================================

ACloudViewer Azure Kinect Viewer is used for previewing RGB and depth image stream
captured by the Azure Kinect sensor.

ACloudViewer provides Python and C++ example code of Azure Kinect viewer. Please
see ``examples/Cpp/AzureKinectViewer.cpp`` and
``examples/Python/reconstruction_system/sensors/azure_kinect_viewer.py``
for details.

We'll use the Python version as an example.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_viewer.py --align_depth_to_color

When the visualizer window is active, press ``ESC`` to quit the viewer.

You may also specify the sensor config with a ``json`` file.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_viewer.py --config config.json

An sensor config will look like the following. For the full list of available
configs, refer to `this file <https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/include/k4a/k4atypes.h>`_.

.. code-block:: json

    {
        "camera_fps" : "K4A_FRAMES_PER_SECOND_30",
        "color_format" : "K4A_IMAGE_FORMAT_COLOR_MJPG",
        "color_resolution" : "K4A_COLOR_RESOLUTION_720P",
        "depth_delay_off_color_usec" : "0",
        "depth_mode" : "K4A_DEPTH_MODE_WFOV_2X2BINNED",
        "disable_streaming_indicator" : "false",
        "subordinate_delay_off_master_usec" : "0",
        "synchronized_images_only" : "false",
        "wired_sync_mode" : "K4A_WIRED_SYNC_MODE_STANDALONE"
    }

ACloudViewer Azure Kinect Recorder
===================================

ACloudViewer Azure Kinect Recorder is used for recording RGB and depth image stream
to an MKV file.

ACloudViewer provides Python and C++ example code of Azure Kinect recorder. Please
see ``examples/Cpp/AzureKinectRecord.cpp`` and
``examples/Python/reconstruction_system/sensors/azure_kinect_recorder.py``
for details.

We'll use the Python version as an example.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_recorder.py --output record.mkv

You may optionally specify the camera config when running the recorder script.

When the visualizer window is active, press ``SPACE`` to start or pause the
recording or press ``ESC`` to quit the recorder.

ACloudViewer Azure Kinect MKV Reader
====================================

The recorded MKV file uses K4A's custom format which contains both RGB and depth
information. To view the customized MKV file, use the
ACloudViewer Azure Kinect MKV Reader.

ACloudViewer provides Python and C++ example code of Azure Kinect MKV Reader.
Please see ``examples/Cpp/AzureKinectMKVReader.cpp`` and
``examples/Python/reconstruction_system/sensors/azure_kinect_mkv_reader.py``
for details.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input record.mkv

To convert the MKV video to color and depth image frames, specify the ``--output``
flag.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input record.mkv --output frames

Python API Example
==================

.. code-block:: python

    import cloudViewer as cv3d
    
    # Create sensor config
    config = cv3d.io.AzureKinectSensorConfig()
    
    # Create sensor
    sensor = cv3d.io.AzureKinectSensor(config)
    
    # Connect to device
    if sensor.connect(0):
        # Capture frame
        rgbd = sensor.capture_frame(enable_align_depth_to_color=True)
        
        # Process RGB-D image
        pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
            rgbd, cv3d.camera.PinholeCameraIntrinsic(
                cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
        
        # Visualize
        cv3d.visualization.draw_geometries([pcd])
        
        sensor.disconnect()

.. seealso::

   - :doc:`../reconstruction_system/capture_your_own_dataset` - Capturing datasets
   - :doc:`../../python_api/cloudViewer.io` - I/O API Reference
