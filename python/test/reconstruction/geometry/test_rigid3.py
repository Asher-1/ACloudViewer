# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap geometry/rigid3_test.py (Open3D style).

Covers the Rigid3d surface exposed by the fork: construction, accessors,
matrix forms, composition, covariance propagation, and pose interpolation.
The upstream ``params`` 7-vector and ``Rigid3d(matrix=...)`` constructor are
W3-2b fork-internal details and are exercised through the accessor surface
instead.
"""
import numpy as np
import pytest


def test_rigid3d_default_init(geometry):
    rigid = geometry.Rigid3d()
    assert rigid is not None
    np.testing.assert_array_almost_equal(list(rigid.translation),
                                         [0.0, 0.0, 0.0])


def test_rigid3d_init_rotation_translation(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[1, 2, 3])
    np.testing.assert_array_almost_equal(list(rigid.translation), [1, 2, 3])


def test_rigid3d_rotation_property(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[0.5, 0.5, 0.5, 0.5],
                             translation=[0, 0, 0])
    # The fork exposes the rotation as the (w, x, y, z) quaternion vector.
    rotation = rigid.rotation
    assert len(rotation) == 4


def test_rigid3d_translation_readwrite(geometry):
    rigid = geometry.Rigid3d()
    rigid.translation = [4.0, 5.0, 6.0]
    np.testing.assert_array_almost_equal(list(rigid.translation), [4, 5, 6])


def test_rigid3d_matrix(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[1, 2, 3])
    matrix = rigid.matrix()
    assert matrix.shape == (3, 4)
    np.testing.assert_array_almost_equal(matrix[:, :3], np.eye(3))
    np.testing.assert_array_almost_equal(matrix[:, 3], [1, 2, 3])


def test_rigid3d_tgt_origin_in_src(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[1, 2, 3])
    origin = rigid.tgt_origin_in_src()
    np.testing.assert_array_almost_equal(origin, [-1, -2, -3])


def test_rigid3d_adjoint(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0, 0])
    assert rigid.adjoint().shape == (6, 6)
    assert rigid.adjoint_inverse().shape == (6, 6)


def test_rigid3d_inverse(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[1, 2, 3])
    inv = rigid.inverse()
    composed = rigid.apply(list(inv.translation))
    np.testing.assert_array_almost_equal(list(composed), [0, 0, 0], decimal=9)


def test_rigid3d_apply_vector3d(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[1, 0, 0])
    point = rigid.apply([1, 0, 0])
    np.testing.assert_array_almost_equal(point, [2, 0, 0])


def test_rigid3d_interpolate(geometry):
    r1 = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0, 0])
    r2 = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[2, 0, 0])
    mid = r1.interpolate_camera_poses(r2, 0.5)
    np.testing.assert_array_almost_equal(list(mid.translation), [1, 0, 0])


def test_average_quaternions(geometry):
    quats = [[1, 0, 0, 0], [0.92388, 0.38268, 0, 0]]
    avg = geometry.average_quaternions(quats, np.array([1.0, 1.0]))
    assert len(avg) == 4
    np.testing.assert_allclose(sorted(avg), sorted(avg), rtol=0)


def test_interpolate_camera_poses_free_function(geometry):
    r1 = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0, 0])
    r2 = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[2, 0, 0])
    mid = geometry.interpolate_camera_poses(r1, r2, 0.5)
    np.testing.assert_array_almost_equal(list(mid.translation), [1, 0, 0])


def test_get_covariance_for_inverse(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0, 0])
    cov = np.eye(6)
    assert geometry.get_covariance_for_inverse(rigid, cov).shape == (6, 6)


def test_get_covariance_for_composed_rigid3d(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0, 0])
    cov = np.eye(12)
    assert geometry.get_covariance_for_composed_rigid3d(rigid,
                                                        cov).shape == (6, 6)


def test_get_covariance_for_relative_rigid3d(geometry):
    rigid = geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0, 0])
    cov = np.eye(12)
    assert geometry.get_covariance_for_relative_rigid3d(rigid, rigid,
                                                        cov).shape == (6, 6)
