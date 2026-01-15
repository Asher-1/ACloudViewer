.. _capture_your_own_dataset:

Capture your own dataset
-------------------------------------

If you have a `RealSense camera
<https://www.intel.com/content/www/us/en/architecture-and-technology/realsense-overview.html>`_
or an Azure Kinect sensor, capturing RGBD frames is easy by using the sensor
recorder scripts.

Input arguments
````````````````````````````````````

For RealSense cameras, use:

.. code-block:: bash

    python examples/Python/reconstruction_system/sensors/realsense_recorder.py --record_imgs

For Azure Kinect sensors, use:

.. code-block:: bash

    python examples/Python/reconstruction_system/sensors/azure_kinect_recorder.py --output record.mkv

The script displays a capturing preview showing the color image and depth map.
Invalid depth pixels are marked (in gray color) and represent object boundaries,
uncertain regions, or distant regions (more than 3m). Capturing frames without
too many gray pixels is recommended for good reconstruction quality.

By default, ``record_imgs`` mode saves aligned color and depth images in
``dataset/realsense`` folder that can be used for reconstruction system.

.. code-block:: bash

    dataset
    └── realsense
        ├── camera_intrinsic.json
        ├── color
        │   ├── 000000.jpg
        │   ├── :
        └── depth
            ├── 000000.png
            ├── :

``camera_intrinsic.json`` has intrinsic parameter of the used camera.
This parameter set should be used with the dataset.

Make a new configuration file
````````````````````````````````````

A new configuration file is required to specify path to the new dataset.
``config/realsense.json`` or ``config/azure_kinect.json`` is provided for this purpose.

Note that ``path_dataset`` and ``path_intrinsic`` indicates paths of dataset
and intrinsic parameters.

Run reconstruction system
````````````````````````````````````

Run the system by using the new configuration file.

.. code-block:: sh

    cd examples/Python/reconstruction_system/
    python run_system.py config/realsense.json [--make] [--register] [--refine] [--integrate]

.. seealso::

   - :doc:`../sensor/azure_kinect` - Azure Kinect sensor
   - :doc:`../sensor/realsense` - RealSense sensor
   - :doc:`system_overview` - System overview
