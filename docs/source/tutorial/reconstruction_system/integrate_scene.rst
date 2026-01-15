.. _reconstruction_system_integrate_scene:

Integrate scene
-------------------------------------

The final step of the scene reconstruction system is to integrate all RGB-D images
to generate a mesh model for the scene.

Input arguments
````````````````````````````````````

The script runs with ``python run_system.py [config] --integrate``. This step
requires that fragments have been refined in the previous step (``--refine``).

TSDF volume integration
````````````````````````````````````

All RGB-D images are integrated into a TSDF (Truncated Signed Distance Function)
volume using the optimized camera poses from the previous steps. The TSDF volume
represents the 3D scene as a signed distance field.

Mesh extraction
````````````````````````````````````

After integration, a triangle mesh is extracted from the TSDF volume using
marching cubes algorithm. The mesh represents the final reconstructed 3D model
of the scene.

Color mapping
````````````````````````````````````

Colors from the RGB images are mapped to the mesh vertices to create a colored
3D model.

The final mesh is saved and can be visualized or used for further processing.

.. seealso::

   - :doc:`system_overview` - System overview
   - :doc:`refine_registration` - Refining registration
   - :doc:`../pipelines/rgbd_integration` - RGB-D integration
