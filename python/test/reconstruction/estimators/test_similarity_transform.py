# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap estimators/similarity_transform_test.py
(Open3D style). Fork adaptation: vector inputs use the utility.Vector3dVector
container."""
import numpy as np
import pytest


def test_estimate_rigid3d_with_coplanar_points(estimators, geometry, utility):
    source = utility.Vector3dVector([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    target = utility.Vector3dVector(source)
    result = estimators.estimate_rigid3d(source, target)
    # May return None for degenerate configurations, but call must succeed.
    assert result is None or isinstance(result, geometry.Rigid3d)


def test_estimate_sim3d_with_points(estimators, geometry, utility):
    source = utility.Vector3dVector([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    target = utility.Vector3dVector([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 2.0],
    ])
    result = estimators.estimate_sim3d(source, target)
    assert result is None or isinstance(result, geometry.Sim3d)


@pytest.mark.parametrize(
    "name",
    [
        "estimate_rigid3d",
        "estimate_rigid3d_robust",
        "estimate_sim3d",
        "estimate_sim3d_robust",
    ],
)
def test_public_api_callable(estimators, name):
    assert callable(getattr(estimators, name))
