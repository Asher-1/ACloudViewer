# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/point3d_test.py (Open3D style)."""


def test_point3d_default(scene):
    p = scene.Point3D()
    assert list(p.xyz) == [0.0, 0.0, 0.0]
    assert not p.has_error()


def test_point3d_xyz_color_error(scene):
    p = scene.Point3D()
    p.xyz = [1.0, 2.0, 3.0]
    p.color = [255, 128, 0]
    p.error = 0.5
    assert list(p.xyz) == [1.0, 2.0, 3.0]
    assert list(p.color) == [255, 128, 0]
    assert p.has_error()
    assert p.error == 0.5


def test_point3d_track(scene):
    p = scene.Point3D()
    t = scene.Track(elements=[scene.TrackElement(1, 0)])
    p.track = t
    assert p.track.length == 1
