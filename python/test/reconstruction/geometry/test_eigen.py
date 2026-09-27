# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap geometry/eigen_test.py (Open3D style).

Covers the Rotation3d and AlignedBox3d wrappers added for parity.
"""
import copy

import numpy as np
import pytest


def test_rotation3d_default_init(geometry):
    r = geometry.Rotation3d()
    assert r is not None


def test_rotation3d_init_xyzw(geometry):
    r = geometry.Rotation3d(xyzw=[0, 0, 0, 1])
    np.testing.assert_allclose(list(r.quat), [0, 0, 0, 1])


def test_rotation3d_init_matrix(geometry):
    r = geometry.Rotation3d(matrix=np.eye(3))
    np.testing.assert_allclose(r.matrix(), np.eye(3), atol=1e-12)


def test_rotation3d_multiply(geometry):
    r1 = geometry.Rotation3d(xyzw=[0, 0, 0, 1])
    r2 = geometry.Rotation3d(xyzw=[0, 0, 0, 1])
    prod = r1 * r2
    assert isinstance(prod, geometry.Rotation3d)
    vec = r1 * np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(vec, [1, 2, 3])


def test_rotation3d_normalize_and_norm(geometry):
    r = geometry.Rotation3d(xyzw=[0, 0, 0, 2])
    r.normalize()
    assert r.norm() == pytest.approx(1.0)


def test_rotation3d_angle_and_inverse(geometry):
    r = geometry.Rotation3d(xyzw=[0, 0, 0, 1])
    assert r.angle() == pytest.approx(0.0)
    inv = r.inverse()
    assert isinstance(inv, geometry.Rotation3d)


def test_rotation3d_angle_to(geometry):
    r1 = geometry.Rotation3d(xyzw=[0, 0, 0, 1])
    r2 = geometry.Rotation3d(xyzw=[0, 0, 0, 1])
    assert r1.angle_to(r2) == pytest.approx(0.0)


def test_rotation3d_deepcopy(geometry):
    r = geometry.Rotation3d(xyzw=[0, 0, 0, 1])
    assert isinstance(copy.deepcopy(r), geometry.Rotation3d)


def test_aligned_box3d_contains(geometry):
    box = geometry.AlignedBox3d(min=[0, 0, 0], max=[1, 2, 3])
    assert box.contains([0.5, 1.0, 1.5])
    assert not box.contains([2.0, 0.0, 0.0])


def test_aligned_box3d_min_max(geometry):
    box = geometry.AlignedBox3d(min=[0, 0, 0], max=[1, 2, 3])
    np.testing.assert_allclose(list(box.min), [0, 0, 0])
    np.testing.assert_allclose(list(box.max), [1, 2, 3])
    box.min = [-1, -1, -1]
    np.testing.assert_allclose(list(box.min), [-1, -1, -1])


def test_aligned_box3d_geometry(geometry):
    box = geometry.AlignedBox3d(min=[0, 0, 0], max=[1, 2, 3])
    np.testing.assert_allclose(list(box.diagonal()), [1, 2, 3])
    np.testing.assert_allclose(list(box.center()), [0.5, 1.0, 1.5])
    assert box.volume() == pytest.approx(6.0)
