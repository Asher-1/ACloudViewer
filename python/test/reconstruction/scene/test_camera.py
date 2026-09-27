"""Port of upstream pycolmap scene/camera_test.py (Open3D style)."""
import numpy as np
import pytest


@pytest.fixture
def pinhole(scene):
    return scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)


def test_camera_model_id_invalid(scene):
    # The fork stores the model id as an integer; an unknown model name
    # raises instead of returning an invalid id.
    with pytest.raises(Exception):
        scene.Camera.create_from_model_name(1, "NOT_A_MODEL", 800.0, 1, 1)


def test_camera_model_id_pinhole(pinhole):
    assert pinhole.model_name == "PINHOLE"


def test_camera_model_id_simple_radial(scene):
    cam = scene.Camera.create_from_model_name(1, "SIMPLE_RADIAL", 800.0,
                                              640, 480)
    assert cam.model_name == "SIMPLE_RADIAL"


def test_camera_create_from_model_name(pinhole):
    assert pinhole.camera_id == 1
    assert pinhole.width == 640
    assert pinhole.height == 480
    # PINHOLE carries fx/fy; the scalar focal_length accessor is reserved
    # for single-focal models (engine CHECK, upstream-identical).
    assert pinhole.focal_length_x == 800.0
    assert pinhole.focal_length_y == 800.0


def test_camera_single_focal_length(scene):
    cam = scene.Camera.create_from_model_name(1, "SIMPLE_RADIAL", 800.0,
                                              640, 480)
    assert cam.focal_length == 800.0


def test_camera_focal_length_xy(pinhole):
    pinhole.focal_length_x = 850.0
    pinhole.focal_length_y = 900.0
    assert pinhole.focal_length_x == 850.0
    assert pinhole.focal_length_y == 900.0


def test_camera_calibration_matrix(pinhole):
    K = pinhole.calibration_matrix()
    assert K.shape == (3, 3)
    assert K[0][0] == pytest.approx(800.0)
    assert K[0][2] == pytest.approx(320.0)
    assert K[1][2] == pytest.approx(240.0)
    assert K[2][2] == pytest.approx(1.0)


def test_camera_cam_from_img_roundtrip(pinhole):
    principal = [320.0, 240.0]
    cam_pt = pinhole.cam_from_img(principal)
    img = pinhole.img_from_cam([cam_pt[0], cam_pt[1], 1.0])
    assert img[0] == pytest.approx(principal[0])
    assert img[1] == pytest.approx(principal[1])


def test_camera_cam_ray_from_img(pinhole):
    ray = pinhole.cam_ray_from_img([320.0, 240.0])
    assert len(ray) == 3


def test_camera_params_readwrite(pinhole):
    params = pinhole.params
    assert len(params) == 4  # PINHOLE: fx, fy, cx, cy
    params[0] = 810.0
    pinhole.params = params
    assert pinhole.focal_length_x == 810.0


def test_camera_params_to_string(pinhole):
    text = pinhole.params_to_string()
    assert "800" in text


def test_camera_set_params_from_string(pinhole):
    pinhole.set_params_from_string("820, 830, 320, 240")
    assert pinhole.focal_length_x == 820.0


def test_camera_verify_and_bogus(pinhole):
    assert pinhole.verify_params()
    assert not pinhole.has_bogus_params(0.5, 2.0, 100.0)


def test_camera_perspective_and_undistorted(pinhole):
    assert pinhole.is_perspective()
    assert pinhole.is_perspective_pinhole()
    assert not pinhole.is_spherical()
    assert pinhole.is_undistorted()


def test_camera_rescale(pinhole):
    pinhole.rescale(2.0)
    assert pinhole.width == 1280
    assert pinhole.focal_length_x == pytest.approx(1600.0)


def test_camera_mean_focal_length(pinhole):
    assert pinhole.mean_focal_length() == pytest.approx(800.0)


def test_camera_idxs(pinhole):
    assert len(pinhole.focal_length_idxs()) == 2
    assert len(pinhole.principal_point_idxs()) == 2
    assert len(pinhole.extra_params_idxs()) == 0
    assert len(pinhole.metadata_params_idxs()) == 0
