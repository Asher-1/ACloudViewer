.. _reconstruction_system_register_fragments:

Register fragments
-------------------------------------

The second step of the scene reconstruction system is to register all fragments
in a global space to detect loop closure.

Input arguments
````````````````````````````````````

The script runs with ``python run_system.py [config] --register``. This step
requires that fragments have been created in the previous step (``--make``).

Global registration
````````````````````````````````````

For each pair of fragments, global registration is performed to find an initial
alignment. The function uses feature-based matching (e.g., FPFH features) to
find correspondences between fragments, then uses RANSAC to estimate a rough
transformation.

ICP refinement
````````````````````````````````````

After global registration, ICP (Iterative Closest Point) is used to refine
the alignment between fragment pairs.

Pose graph optimization
````````````````````````````````````

A pose graph is built where each node represents a fragment and edges represent
the transformations between fragments. Loop closures are detected and added to
the pose graph. Global optimization is then performed to optimize all fragment
poses simultaneously.

The optimized pose graph is saved and can be used in the next step.

.. seealso::

   - :doc:`system_overview` - System overview
   - :doc:`make_fragments` - Making fragments
   - :doc:`refine_registration` - Refining registration

- :doc:`../pipelines/global_registration` - Global registration
- :doc:`../pipelines/icp_registration` - ICP registration
