# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap geometry/essential_matrix_test.py (Open3D
style)."""
import numpy as np


def test_essential_matrix_from_pose(geometry):
    # Fork adaptation: Rigid3d takes a [w, x, y, z] quaternion array.
    identity_quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    translation = np.array([1.0, 0.0, 0.0])
    rigid = geometry.Rigid3d(identity_quaternion, translation)
    essential_matrix = geometry.essential_matrix_from_pose(rigid)
    assert essential_matrix.shape == (3, 3)
