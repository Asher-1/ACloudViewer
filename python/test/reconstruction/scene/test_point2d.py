# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap scene/point2d_test.py (Open3D style)."""


def test_point2d_default(scene):
    p = scene.Point2D()
    assert p.x() == 0.0 and p.y() == 0.0
    assert not p.has_point3D()


def test_point2d_xy_accessors(scene):
    p = scene.Point2D()
    p.xy = [1.5, 2.5]
    assert p.x() == 1.5
    assert p.y() == 2.5
    assert list(p.xy) == [1.5, 2.5]


def test_point2d_point3d_id(scene):
    p = scene.Point2D(xy=[100.0, 200.0], point3D_id=7)
    assert p.has_point3D()
    assert p.point3D_id == 7
    p.point3D_id = 2**64 - 1  # kInvalidPoint3DId (point3D_t = uint64)
    assert not p.has_point3D()
