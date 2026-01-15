.. _reconstruction_system_refine_registration:

Refine registration
-------------------------------------

The third step of the scene reconstruction system is to refine the rough
alignments obtained from the previous step.

Input arguments
````````````````````````````````````

The script runs with ``python run_system.py [config] --refine``. This step
requires that fragments have been registered in the previous step (``--register``).

Point-to-plane ICP
````````````````````````````````````

Point-to-plane ICP is used to refine the alignments. This method is more
accurate than point-to-point ICP as it takes into account surface normals.

Multiway registration refinement
````````````````````````````````````

After individual pair-wise refinement, multiway registration is performed again
to ensure global consistency across all fragments.

The refined pose graph is saved and can be used in the final integration step.

.. seealso::

   - :doc:`system_overview` - System overview
   - :doc:`register_fragments` - Registering fragments
   - :doc:`integrate_scene` - Integrating scene

- :doc:`../pipelines/icp_registration` - ICP registration
