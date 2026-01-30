.. _azure_kinect:

Azure Kinect with CloudViewer
-------------------------------

Azure Kinect is officially supported on Windows and Ubuntu 20.04+.

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

Currently, CloudViewer supports the Azure Kinect SDK version ``v1.4.1``, though future
versions might also be compatible.

Install CloudViewer from Pip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're using CloudViewer installed via Pip, CloudViewer's Azure Kinect features
should work out-of-the box if K4A is installed in the system in the recommended
way. CloudViewer will try to load the K4A dynamic library automatically at runtime,
when a K4A related feature within CloudViewer is used.

On Ubuntu, the default search path
follows the Linux `convention <https://unix.stackexchange.com/a/22999/130082>`_.

On Windows, CloudViewer will try to load the shared library from the default
installation path. For example, for K4A ``v1.4.1``, the default path is
``C:\Program Files\Azure Kinect SDK v1.4.1``. If this doesn't work, copy
``depthengine_x_x.dll``, ``k4a.dll`` and ``k4arecord.dll`` to where CloudViewer
Python module is installed if you're using CloudViewer with Python, or to the same
directory as your C++ executable.

You can get CloudViewer's Python module path with the following command:

.. code-block:: sh

    python -c "import cloudViewer as cv3d; import os; print(os.path.dirname(cv3d.__file__))"

Compile from Source
~~~~~~~~~~~~~~~~~~~

To build CloudViewer from source with K4A support, set ``BUILD_AZURE_KINECT=ON`` at
CMake config step. That is,

.. code-block:: sh

    cmake -DBUILD_AZURE_KINECT=ON -DOTHER_FLAGS ..

CloudViewer Azure Kinect Viewer
=================================

CloudViewer Azure Kinect Viewer is used for previewing RGB and depth image stream
captured by the Azure Kinect sensor.

CloudViewer provides Python and C++ example code of Azure Kinect viewer. Please
see ``examples/Cpp/AzureKinectViewer.cpp`` and
``examples/Python/reconstruction_system/sensors/azure_kinect_viewer.py``
for details.

We'll use the Python version as an example.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_viewer.py --align_depth_to_color

When recording at a higher resolution at a high framerate, sometimes it is
helpful to use the raw depth image without transformation to reduce computation.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_viewer.py

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

CloudViewer Azure Kinect Recorder
===================================

CloudViewer Azure Kinect Recorder is used for recording RGB and depth image stream
to an MKV file.

CloudViewer provides Python and C++ example code of Azure Kinect recorder. Please
see ``examples/Cpp/AzureKinectRecord.cpp`` and
``examples/Python/reconstruction_system/sensors/azure_kinect_recorder.py``
for details.

We'll use the Python version as an example.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_recorder.py --output record.mkv

You may optionally specify the camera config when running the recorder script.

When the visualizer window is active, press ``SPACE`` to start or pause the
recording or press ``ESC`` to quit the recorder.

CloudViewer Azure Kinect MKV Reader
====================================

The recorded MKV file uses K4A's custom format which contains both RGB and depth
information. The regular video player may only support playing back the color channel
or not supporting the format at all. To view the customized MKV file, use the
CloudViewer Azure Kinect MKV Reader.

CloudViewer provides Python and C++ example code of CloudViewer Azure Kinect MKV Reader.
Please see ``examples/Cpp/AzureKinectMKVReader.cpp`` and
``examples/Python/reconstruction_system/sensors/azure_kinect_mkv_reader.py``
for details.

.. code-block:: sh

    python examples/Python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input record.mkv

Note that even though the recorder records the unaligned raw depth image, the
reader can correctly wrap the depth image to align with the color image.

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
