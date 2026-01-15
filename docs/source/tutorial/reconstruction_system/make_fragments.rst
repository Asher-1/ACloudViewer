.. _reconstruction_system_make_fragments:

Make fragments
-------------------------------------

The first step of the scene reconstruction system is to create fragments from
short RGBD sequences.

Input arguments
````````````````````````````````````

The script runs with ``python run_system.py [config] --make``. In ``[config]``,
``["path_dataset"]`` should have subfolders ``image`` and ``depth`` to store the
color images and depth images respectively. We assume the color images and the
depth images are synchronized and registered. In ``[config]``, the optional
argument ``["path_intrinsic"]`` specifies the path to a json file that stores
the camera intrinsic matrix. If it is not given, the PrimeSense factory setting is used instead.

Register RGBD image pairs
````````````````````````````````````

The function reads a pair of RGBD images and registers the ``source_rgbd_image``
to the ``target_rgbd_image``. The ACloudViewer function ``compute_rgbd_odometry`` is
called to align the RGBD images. For adjacent RGBD images, an identity matrix is
used as the initialization. For non-adjacent RGBD images, wide baseline matching
is used as the initialization.

Multiway registration
````````````````````````````````````

This script uses the technique demonstrated in
:ref:`/tutorial/pipelines/multiway_registration.ipynb`. The function
``make_posegraph_for_fragment`` builds a pose graph for multiway registration of
all RGBD images in this sequence. Each graph node represents an RGBD image and
its pose which transforms the geometry to the global fragment space.
For efficiency, only key frames are used.

Once a pose graph is created, multiway registration is performed by calling the
function ``optimize_posegraph_for_fragment``.

This function calls ``global_optimization`` to estimate poses of the RGBD images.

Make a fragment
````````````````````````````````````

After pose graph optimization, RGBD images are integrated into a TSDF volume
to create a fragment. The function ``integrate_rgb_frames_for_fragment``
performs this integration.

Each fragment is saved as a point cloud file that can be used in the next step.

.. seealso::

   - :doc:`system_overview` - System overview
   - :doc:`register_fragments` - Registering fragments
   - :doc:`../pipelines/rgbd_odometry` - RGB-D odometry
   - :doc:`../pipelines/rgbd_integration` - RGB-D integration
