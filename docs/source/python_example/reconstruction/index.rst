Reconstruction Examples
========================

Python examples for 3D reconstruction using COLMAP-based structure-from-motion and multi-view stereo.

Overview
--------

The reconstruction module provides tools for:

- **Structure-from-Motion (SfM)**: Recover camera poses and sparse 3D points
- **Multi-View Stereo (MVS)**: Generate dense point clouds from images
- **Feature Detection and Matching**: Extract and match features across images
- **Database Management**: Store and query image features and matches
- **Vocabulary Tree**: Fast image retrieval for loop closure

Example Code
------------

For complete runnable examples, see:

**Structure-from-Motion:**
- `sfm.py <../../../examples/Python/reconstruction/sfm.py>`_ - Structure-from-motion pipeline

**Multi-View Stereo:**
- `mvs.py <../../../examples/Python/reconstruction/mvs.py>`_ - Multi-view stereo reconstruction

**Feature Detection:**
- `feature.py <../../../examples/Python/reconstruction/feature.py>`_ - Feature extraction and matching

**Database Management:**
- `database.py <../../../examples/Python/reconstruction/database.py>`_ - Database operations

**Image Processing:**
- `image.py <../../../examples/Python/reconstruction/image.py>`_ - Image handling

**Model Management:**
- `model.py <../../../examples/Python/reconstruction/model.py>`_ - 3D model operations

**Vocabulary Tree:**
- `vocab_tree.py <../../../examples/Python/reconstruction/vocab_tree.py>`_ - Vocabulary tree for image retrieval

**GUI:**
- `gui.py <../../../examples/Python/reconstruction/gui.py>`_ - Graphical user interface

See Also
--------

- :doc:`../../tutorial/reconstruction_system/index` - Reconstruction System Tutorials
- :doc:`../../python_api/cloudViewer.reconstruction` - Reconstruction API Reference
