# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap scene/constants_test.py (Open3D style).

The fork exposes the invalid-id sentinels through the typed surfaces.
"""


def test_invalid_image_id(scene):
    img = scene.Image()
    assert img.image_id == 2**32 - 1


def test_invalid_point3d_id(scene):
    p = scene.Point2D()
    p.point3D_id = 2**64 - 1
    assert not p.has_point3D()


def test_invalid_camera_id(scene):
    cam = scene.Camera()
    assert cam.camera_id == 2**32 - 1
