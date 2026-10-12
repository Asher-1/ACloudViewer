# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/image_test.py (Open3D style)."""
import numpy as np


def test_image_default(scene):
    img = scene.Image()
    assert img.name == ""
    assert img.image_id == 2**32 - 1
    assert img.num_points2D == 0


def test_image_name_camera_id(scene):
    img = scene.Image()
    img.name = "img1.jpg"
    img.camera_id = 1
    assert img.name == "img1.jpg"
    assert img.camera_id == 1


def test_image_project_point(scene, geometry):
    """project_point via the trivial-pose composition (batch 10 parity)."""
    import pytest
    rec = scene.Reconstruction()
    cam = scene.Camera.create_from_model_name(1, "PINHOLE", 800.0, 640, 480)
    rec.add_camera_with_trivial_rig(cam)
    img = scene.Image()
    img.name = "a.jpg"
    img.camera_id = 1
    img.image_id = 1
    rec.add_image_with_trivial_frame(
        img, geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0,
                                                                       0]))
    uv = rec.image(1).project_point([0.0, 0.0, 5.0])
    assert uv[0] == pytest.approx(320.0)
    assert uv[1] == pytest.approx(240.0)
