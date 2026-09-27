# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap estimators/triangulation_test.py."""
import numpy as np


def test_estimate_triangulation(scene, estimators, geometry):
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    cam_from_world_l = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0],
                                        translation=[-1.0, 0.0, 0.0])
    cam_from_world_r = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0],
                                        translation=[1.0, 0.0, 0.0])
    opts = estimators.EstimateTriangulationOptions()
    # Two horizontally displaced views of a point in front of both.
    point = opts.estimate_triangulation(
        np.array([[160.0, 240.0], [480.0, 240.0]]),
        [cam_from_world_l, cam_from_world_r], [cam, cam])
    assert point is not None and len(point) >= 2
