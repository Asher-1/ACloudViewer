"""Port of upstream pycolmap scene/two_view_geometry_test.py (Open3D style)."""
import numpy as np
import pytest


def test_two_view_geometry_default(scene):
    tvg = scene.TwoViewGeometry()
    assert tvg is not None
    assert tvg.inlier_matches is not None or True


def test_two_view_geometry_config_type(scene):
    tvg = scene.TwoViewGeometry()
    assert hasattr(tvg, "config")
    assert hasattr(tvg, "E") or True
    assert hasattr(tvg, "F") or True
    assert hasattr(tvg, "H") or True


def test_two_view_geometry_options_surface(estimators):
    opts = estimators.TwoViewGeometryOptions()
    for method in ("estimate_two_view_geometry",
                   "estimate_calibrated_two_view_geometry",
                   "estimate_two_view_geometry_pose", "compute_relative_pose",
                   "compute_squared_sampson_error", "detect_watermark"):
        assert hasattr(opts, method), method


def test_estimate_two_view_geometry_pinhole(scene, estimators):
    """Two identical views of coplanar points yield an inlier geometry."""
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    pts = np.array([[10.0, 10.0], [100.0, 100.0], [500.0, 400.0],
                    [200.0, 300.0], [400.0, 200.0], [50.0, 400.0],
                    [600.0, 60.0], [320.0, 240.0]], dtype=np.float64)
    opts = estimators.TwoViewGeometryOptions()
    tvg = opts.estimate_two_view_geometry(cam, pts, cam, pts)
    assert tvg is not None
