# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap estimators/cost_functions_test.py (Open3D style).

The upstream only asserts that the factory entry points exist; this port
additionally invokes every overload and checks the residual/parameter-block
layout through the reconstruction.pyceres.CostFunction surface, which the
fork registers so the factories are usable without pyceres.
"""
import numpy as np
import pytest


@pytest.fixture
def cost_functions(recon):
    return recon.cost_functions


@pytest.fixture
def model_id(scene):
    return scene.CameraModelId.PINHOLE


@pytest.fixture
def pose(geometry):
    return geometry.Rigid3d(rotation_wxyz=[1, 0, 0, 0], translation=[0, 0, 0])


def test_cost_functions_submodule_exists(recon):
    assert hasattr(recon, "cost_functions")


def test_reproj_error_cost_exists(cost_functions):
    assert hasattr(cost_functions, "ReprojErrorCost")


def test_rig_reproj_error_cost_exists(cost_functions):
    assert hasattr(cost_functions, "RigReprojErrorCost")


def test_scaled_rig_reproj_error_cost_exists(cost_functions):
    assert hasattr(cost_functions, "ScaledRigReprojErrorCost")


def test_sampson_error_cost_exists(cost_functions):
    assert hasattr(cost_functions, "SampsonErrorCost")


def test_absolute_pose_prior_cost_exists(cost_functions):
    assert hasattr(cost_functions, "AbsolutePosePriorCost")


def test_absolute_pose_position_prior_cost_exists(cost_functions):
    assert hasattr(cost_functions, "AbsolutePosePositionPriorCost")


def test_relative_pose_prior_cost_exists(cost_functions):
    assert hasattr(cost_functions, "RelativePosePriorCost")


def test_point3d_alignment_cost_exists(cost_functions):
    assert hasattr(cost_functions, "Point3DAlignmentCost")


# --- Behavioral checks over the factory surface ---------------------------


def test_reproj_error_cost_layout(cost_functions, model_id):
    cf = cost_functions.ReprojErrorCost(camera_model_id=model_id,
                                        point2D=[320.0, 240.0])
    assert cf.num_residuals() == 2
    # point3D, cam_from_world, camera params (PINHOLE has 4).
    assert list(cf.parameter_block_sizes()) == [3, 7, 4]


def test_reproj_error_cost_weighted_variants(cost_functions, model_id):
    cov = [[1.0, 0.0], [0.0, 1.0]]
    for kwargs in (
        {
            "point2D_cov": cov,
            "point2D": [0.0, 0.0]
        },
        {
            "point2D_stddev": 0.5,
            "point2D": [0.0, 0.0]
        },
        {
            "point2D_stddev": [0.5, 0.6],
            "point2D": [0.0, 0.0]
        },
    ):
        cf = cost_functions.ReprojErrorCost(camera_model_id=model_id, **kwargs)
        assert cf.num_residuals() == 2


def test_reproj_error_cost_constant_pose_and_point(cost_functions, model_id,
                                                   pose):
    cf = cost_functions.ReprojErrorCost(camera_model_id=model_id,
                                        point2D=[0.0, 0.0],
                                        cam_from_world=pose)
    # The pose is folded into the functor: only point3D and camera remain.
    assert list(cf.parameter_block_sizes()) == [3, 4]
    cf = cost_functions.ReprojErrorCost(camera_model_id=model_id,
                                        point2D=[0.0, 0.0],
                                        point3D=[0.0, 0.0, 5.0])
    assert list(cf.parameter_block_sizes()) == [7, 4]


def test_rig_reproj_error_cost_layout(cost_functions, model_id, pose):
    cf = cost_functions.RigReprojErrorCost(camera_model_id=model_id,
                                           point2D=[0.0, 0.0])
    # point3D, cam_from_rig, rig_from_world, camera.
    assert list(cf.parameter_block_sizes()) == [3, 7, 7, 4]
    cf = cost_functions.RigReprojErrorCost(camera_model_id=model_id,
                                           point2D=[0.0, 0.0],
                                           cam_from_rig=pose)
    assert list(cf.parameter_block_sizes()) == [3, 7, 4]
    sf = cost_functions.ScaledRigReprojErrorCost(camera_model_id=model_id,
                                                 point2D=[0.0, 0.0],
                                                 use_log_scale=True)
    # point3D, cam_from_rig, scaled rig_from_world (8), camera.
    assert list(sf.parameter_block_sizes()) == [3, 7, 8, 4]


def test_sampson_error_cost_layout(cost_functions):
    cf = cost_functions.SampsonErrorCost(point1=[0.0, 0.0], point2=[1.0, 1.0])
    assert cf.num_residuals() == 1
    assert list(cf.parameter_block_sizes()) == [7]


def test_absolute_pose_prior_cost_layout(cost_functions, pose):
    cf = cost_functions.AbsolutePosePriorCost(cam_from_world_prior=pose)
    assert cf.num_residuals() == 6
    assert list(cf.parameter_block_sizes()) == [7]
    cov = np.eye(6)
    cf = cost_functions.AbsolutePosePriorCost(cam_cov_from_world_prior=cov,
                                              cam_from_world_prior=pose)
    assert cf.num_residuals() == 6
    cf = cost_functions.AbsolutePosePriorCost(
        cam_stddev_from_world_prior=[0.1] * 6, cam_from_world_prior=pose)
    assert cf.num_residuals() == 6


def test_absolute_pose_position_prior_cost_layout(cost_functions):
    cf = cost_functions.AbsolutePosePositionPriorCost(
        position_in_world_prior=[0.0, 0.0, 0.0])
    assert cf.num_residuals() == 3
    assert list(cf.parameter_block_sizes()) == [7]
    cf = cost_functions.AbsolutePosePositionPriorCost(
        position_cov_in_world_prior=np.eye(3),
        position_in_world_prior=[0.0, 0.0, 0.0])
    assert cf.num_residuals() == 3
    cf = cost_functions.AbsolutePosePositionPriorCost(
        position_stddev_in_world_prior=0.1,
        position_in_world_prior=[0.0, 0.0, 0.0])
    assert cf.num_residuals() == 3
    cf = cost_functions.AbsolutePosePositionPriorCost(
        position_stddev_in_world_prior=[0.1, 0.2, 0.3],
        position_in_world_prior=[0.0, 0.0, 0.0])
    assert cf.num_residuals() == 3


def test_relative_pose_prior_cost_layout(cost_functions, pose):
    cf = cost_functions.RelativePosePriorCost(i_from_j_prior=pose)
    assert cf.num_residuals() == 6
    assert list(cf.parameter_block_sizes()) == [7, 7]


def test_point3d_alignment_cost_layout(cost_functions):
    cf = cost_functions.Point3DAlignmentCost(point_in_b_prior=[1.0, 2.0, 3.0],
                                             use_log_scale=True)
    assert cf.num_residuals() == 3
    # point_in_a, b_from_a similarity (quat + translation + log-scale).
    assert list(cf.parameter_block_sizes()) == [3, 8]
