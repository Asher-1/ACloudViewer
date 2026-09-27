# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap geometry/pose_prior_test.py (Open3D style)."""
import numpy as np


def test_pose_prior_coordinate_system_enum(geometry):
    assert geometry.PosePriorCoordinateSystem.UNDEFINED is not None
    assert geometry.PosePriorCoordinateSystem.WGS84 is not None
    assert geometry.PosePriorCoordinateSystem.CARTESIAN is not None


def test_pose_prior_default_init(geometry):
    assert geometry.PosePrior() is not None


def test_pose_prior_position_readwrite(geometry):
    pose_prior = geometry.PosePrior()
    pose_prior.position = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_almost_equal(pose_prior.position, [1.0, 2.0, 3.0])


def test_pose_prior_position_covariance_readwrite(geometry):
    pose_prior = geometry.PosePrior()
    covariance = np.eye(3) * 0.5
    pose_prior.position_covariance = covariance
    np.testing.assert_array_almost_equal(pose_prior.position_covariance,
                                         covariance)


def test_pose_prior_coordinate_system_readwrite(geometry):
    pose_prior = geometry.PosePrior()
    pose_prior.coordinate_system = geometry.PosePriorCoordinateSystem.WGS84
    assert (pose_prior.coordinate_system ==
            geometry.PosePriorCoordinateSystem.WGS84)


def test_pose_prior_gravity_readwrite(geometry):
    pose_prior = geometry.PosePrior()
    gravity = np.array([0.0, 0.0, -9.81])
    pose_prior.gravity = gravity
    np.testing.assert_array_almost_equal(pose_prior.gravity, gravity)
