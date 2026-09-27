# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap estimators/essential_matrix_test.py (Open3D
style): the estimator surface lives on recon.estimators."""


def test_estimate_essential_matrix_is_callable(estimators):
    assert callable(estimators.estimate_essential_matrix)
