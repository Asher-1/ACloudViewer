.. _realsense:

RealSense with ACloudViewer
===========================

RealSense (``librealsense`` SDK v2) is integrated into ACloudViewer and you
can use it through both C++ and Python APIs. RealSense support is available
on Linux, macOS and Windows.

Obtaining ACloudViewer with RealSense support
---------------------------------------------

Compile from source (C++)
^^^^^^^^^^^^^^^^^^^^^^^^^
To build ACloudViewer from source with RealSense support, set
``BUILD_LIBREALSENSE=ON`` at CMake config step. You can add other configuration
options as well (e.g.: ``BUILD_GUI=ON`` and ``BUILD_PYTHON_MODULE=ON`` may be
useful).

.. code-block:: sh

    cmake -D BUILD_LIBREALSENSE=ON -D <OTHER_FLAGS> /path/to/ACloudViewer/source/

Reading from RealSense bag files
---------------------------------

Here is a C++ code snippet that shows how to read a RealSense bag file recorded
with ACloudViewer or the ``realsense-viewer``. Note that general ROSbag files are not
supported.

.. code-block:: C++

    #include <cloudViewer/CloudViewer.h>
    using namespace cloudViewer;
    t::io::RSBagReader bag_reader;
    bag_reader.Open(bag_filename);
    auto im_rgbd = bag_reader.NextFrame();
    while (!bag_reader.IsEOF()) {
        // process im_rgbd.depth_ and im_rgbd.color_
        im_rgbd = bag_reader.NextFrame();
    }
    bag_reader.Close();

Here is the corresponding Python code:

.. code-block:: Python

    import cloudViewer as cv3d
    bag_reader = cv3d.t.io.RSBagReader()
    bag_reader.open(bag_filename)
    im_rgbd = bag_reader.next_frame()
    while not bag_reader.is_eof():
        # process im_rgbd.depth and im_rgbd.color
        im_rgbd = bag_reader.next_frame()
    
    bag_reader.close()

RealSense camera configuration, live capture, processing and recording
----------------------------------------------------------------------

RealSense camera discovery
^^^^^^^^^^^^^^^^^^^^^^^^^^

You can list all RealSense devices connected to the system and get their
capabilities (supported resolutions, frame rates, etc.) with the code snippet
below.

.. code-block:: C++

    #include <cloudViewer/CloudViewer.h>
    cloudViewer::t::io::RealSenseSensor::ListDevices();

.. code-block:: Python

    import cloudViewer as cv3d
    cv3d.t.io.RealSenseSensor.list_devices()

RealSense camera configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RealSense cameras can be configured with a simple ``json`` configuration file.
See `RealSense documentation
<https://intelrealsense.github.io/librealsense/doxygen/rs__option_8h.html>`_ for
the set of configuration values. Supported configuration options will depend
on the device and other chosen options. Here are the options supported by
ACloudViewer:

* **serial**: Pick a specific device, leave empty to pick the first available
  device.
* **color_format**:  Pixel format for color frames.
* **color_resolution**: (width, height): Leave 0 to let RealSense pick a
  supported width or height.
* **depth_format**: Pixel format for depth frames.
* **depth_resolution**: (width, height): Leave 0 to let RealSense pick a
  supported width or height.
* **fps**: Common frame rate for both depth and color streams. Leave 0 to let
  RealSense pick a supported frame rate.
* **visual_preset**: Controls depth computation on the device. Supported values
  are specific to product line (SR300, RS400, L500). Leave empty to pick the
  default.

Here is an example ``json`` configuration file to capture 30fps, 540p color and
480p depth video from any connected RealSense camera:

.. code-block:: json

  {
      "serial": "",
      "color_format": "RS2_FORMAT_RGB8",
      "color_resolution": "0,540",
      "depth_format": "RS2_FORMAT_Z16",
      "depth_resolution": "0,480",
      "fps": "30",
      "visual_preset": "RS2_L500_VISUAL_PRESET_MAX_RANGE"
   }

RealSense camera capture, processing and recording
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following code snippets show how to capture live RGBD video from a RealSense
camera. They capture the first 150 frames and also record them to an RS bag
file.

.. code-block:: C++

    #include <cloudViewer/CloudViewer.h>
    cloudViewer::t::io::RealSenseSensorConfig rs_cfg;
    cloudViewer::io::ReadIJsonConvertible(config_filename, rs_cfg);
    RealSenseSensor rs;
    rs.InitSensor(rs_cfg, 0, bag_filename);
    rs.StartCapture(true);  // true: start recording with capture
    for(size_t fid = 0; fid<150; ++fid) {
        im_rgbd = rs.CaptureFrame(true, true);  // wait for frames and align them
        // process im_rgbd.depth_ and im_rgbd.color_
    }
    rs.StopCapture();

.. code-block:: Python

    import json
    import cloudViewer as cv3d
    with open(config_filename) as cf:
        rs_cfg = cv3d.t.io.RealSenseSensorConfig(json.load(cf))
    
    rs = cv3d.t.io.RealSenseSensor()
    rs.init_sensor(rs_cfg, 0, bag_filename)
    rs.start_capture(True)  # true: start recording with capture
    for fid in range(150):
        im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
        # process im_rgbd.depth and im_rgbd.color
    
    rs.stop_capture()

Example Code
^^^^^^^^^^^^

For complete examples, see:
- `examples/Python/reconstruction_system/sensors/realsense_recorder.py <../../../examples/Python/reconstruction_system/sensors/realsense_recorder.py>`_
- `examples/Python/reconstruction_system/sensors/realsense_pcd_visualizer.py <../../../examples/Python/reconstruction_system/sensors/realsense_pcd_visualizer.py>`_
- `examples/Python/io/realsense_io.py <../../../examples/Python/io/realsense_io.py>`_

.. seealso::

   - :doc:`../reconstruction_system/capture_your_own_dataset` - Capturing datasets
   - :doc:`../../python_api/cloudViewer.t.io` - Tensor I/O API Reference
