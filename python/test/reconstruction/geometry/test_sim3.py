# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap geometry/sim3_test.py (Open3D style)."""
import pytest
import numpy as np


def test_sim3d_default_init(geometry):
    sim = geometry.Sim3d()
    assert sim is not None
    if hasattr(sim, 'scale'):
        assert sim.scale == pytest.approx(1.0)


def test_sim3d_scale_readwrite(geometry):
    sim = geometry.Sim3d(scale=2.0,
                         rotation_wxyz=[1, 0, 0, 0],
                         translation=[1, 0, 0])
    assert sim.scale == 2.0
    sim.scale = 3.0
    assert sim.scale == 3.0


def test_sim3d_transform_camera_world(geometry):
    sim = geometry.Sim3d(scale=2.0,
                         rotation_wxyz=[1, 0, 0, 0],
                         translation=[1, 0, 0])
    cam = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[1, 1, 1])
    transformed = sim.transform_camera_world(cam)
    # cam_from_new = cam_from_world o Inv(new_from_old); identical to the
    # upstream implementation.
    np.testing.assert_allclose(list(transformed.translation), [1, 2, 2])


def test_sim3d_inverse(geometry):
    sim = geometry.Sim3d(scale=2.0,
                         rotation_wxyz=[1, 0, 0, 0],
                         translation=[1, 0, 0])
    assert sim.inverse() is not None


def test_sim3d_matrix(geometry):
    sim = geometry.Sim3d(scale=2.0,
                         rotation_wxyz=[1, 0, 0, 0],
                         translation=[1, 0, 0])
    assert sim.matrix().shape == (3, 4)
