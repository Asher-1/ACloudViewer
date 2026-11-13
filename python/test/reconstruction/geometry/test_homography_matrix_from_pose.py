# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap geometry/homography_matrix_test.py (Open3D
style)."""


def test_pose_from_homography_matrix_exists(geometry):
    assert hasattr(geometry, "pose_from_homography_matrix")
    assert callable(geometry.pose_from_homography_matrix)
