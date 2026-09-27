"""Port of upstream pycolmap image/undistortion_test.py (Open3D style)."""


def test_undistort_camera_options_default(image):
    opts = image.UndistortCameraOptions()
    assert opts.blank_pixels == 0.0
    opts.min_scale = 0.5
    assert opts.min_scale == 0.5


def test_undistort_image_entry_point(image):
    cam = image.UndistortCameraOptions()
    assert hasattr(cam, "undistort_camera")
    assert hasattr(cam, "undistort_image")
