# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap estimators/homography_matrix_test.py (Open3D
style)."""


def test_estimate_homography_matrix_is_callable(estimators):
    assert callable(estimators.estimate_homography_matrix)
