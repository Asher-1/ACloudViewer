.. _reconstruction_system_overview:

System overview
-----------------------------------

The reconstruction system has 4 main steps:

**Step 1**. :ref:`reconstruction_system_make_fragments`: build local geometric
surfaces (referred to as
fragments) from short subsequences of the input RGBD sequence. This part uses
:ref:`/tutorial/pipelines/rgbd_odometry.ipynb`,
:ref:`/tutorial/pipelines/multiway_registration.ipynb`, and
:ref:`/tutorial/pipelines/rgbd_integration.ipynb`.

**Step 2**. :ref:`reconstruction_system_register_fragments`: the fragments are
aligned in a global space to detect loop closure. This part uses
:ref:`/tutorial/pipelines/global_registration.ipynb`,
:ref:`/tutorial/pipelines/icp_registration.ipynb`, and
:ref:`/tutorial/pipelines/multiway_registration.ipynb`.

**Step 3**. :ref:`reconstruction_system_refine_registration`: the rough
alignments are aligned more tightly. This part uses
:ref:`/tutorial/pipelines/icp_registration.ipynb`, and
:ref:`/tutorial/pipelines/multiway_registration.ipynb`.

**Step 4**. :ref:`reconstruction_system_integrate_scene`: integrate RGB-D images
to generate a mesh model for
the scene. This part uses :ref:`/tutorial/pipelines/rgbd_integration.ipynb`.

.. _reconstruction_system_dataset:

Example dataset
````````````````````````````````````

We provide default datasets such as Lounge RGB-D dataset from Stanford, Bedroom RGB-D dataset from Redwood,
to demonstrate the system in this tutorial.
Other than this, one may use any RGB-D data.
There are lots of excellent RGBD datasets such as: 
`Redwood data <http://redwood-data.org/>`_, `TUM RGBD data <https://vision.in.tum.de/data/datasets/rgbd-dataset>`_, 
`ICL-NUIM data <https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html>`_, 
`the SceneNN dataset <http://people.sutd.edu.sg/~saikit/projects/sceneNN/>`_ and `SUN3D data <http://sun3d.cs.princeton.edu/>`_.

.. _reconstruction_system_how_to_run_the_pipeline:

Quick start
````````````````````````````````````
Getting the example code

.. code-block:: sh

    # Activate your conda environment, where you have installed cloudViewer pip package.
    # Clone the ACloudViewer github repository and go to the example.
    cd examples/Python/reconstruction_system/

    # Show CLI help for `run_system.py`
    python run_system.py --help

Running the example with default dataset.

.. code-block:: sh

    # The following command, will download and use the default dataset,
    # which is ``lounge`` dataset from stanford. 
    # --make will make fragments from RGBD sequence.
    # --register will register all fragments to detect loop closure.
    # --refine flag will refine rough registrations.
    # --integrate flag will integrate the whole RGBD sequence to make final mesh.
    python run_system.py --make --register --refine --integrate

Changing the default dataset.
One may change the default dataset to other available datasets.
Currently the following datasets are available:

1. Lounge (keyword: ``lounge``) (Default)

2. Bedroom (keyword: ``bedroom``)

.. seealso::

   - :doc:`make_fragments` - Making fragments
   - :doc:`register_fragments` - Registering fragments
   - :doc:`refine_registration` - Refining registration
   - :doc:`integrate_scene` - Integrating scene
   - :doc:`capture_your_own_dataset` - Capturing your own dataset
