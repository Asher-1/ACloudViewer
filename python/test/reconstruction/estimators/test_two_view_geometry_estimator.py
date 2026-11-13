# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap estimators/two_view_geometry_test.py."""
import numpy as np


def test_compute_squared_sampson_error(scene, estimators):
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    pts = np.array([[10.0, 10.0], [100.0, 100.0], [500.0, 400.0]])
    opts = estimators.TwoViewGeometryOptions()
    errs = opts.compute_squared_sampson_error(pts, pts, np.eye(3))
    assert len(errs) == 3
    assert np.all(np.asarray(errs) >= 0.0)


def test_estimate_two_view_geometry_identical_views(scene, estimators):
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    pts = np.array([[10.0, 10.0], [100.0, 100.0], [500.0,
                                                   400.0], [200.0, 300.0],
                    [400.0, 200.0], [50.0, 400.0], [600.0, 60.0],
                    [320.0, 240.0]])
    opts = estimators.TwoViewGeometryOptions()
    tvg = opts.estimate_two_view_geometry(cam, pts, cam, pts)
    assert tvg is not None
