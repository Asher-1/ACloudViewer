# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap estimators/motion_averaging_test.py and
rotation_averaging_ceres_test.py (Open3D style)."""
import gc

import numpy as np


def test_rotation_weight_type_enum(estimators):
    assert {
        k: int(v) for k, v in estimators.RotationWeightType.__members__.items()
    } == {
        "GEMAN_MCCLURE": 0,
        "HALF_NORM": 1,
    }


def test_rotation_averaging_reweighting_enum(estimators):
    assert {
        k: int(v)
        for k, v in estimators.RotationAveragingReweighting.__members__.items()
    } == {
        "UNIFORM": 0,
        "INLIER_MATCH_COUNT": 1
    }


def test_rotation_averaging_backend_enum(estimators):
    assert {
        k: int(v)
        for k, v in estimators.RotationAveragingBackend.__members__.items()
    } == {
        "L1_IRLS": 0,
        "CERES": 1,
    }


def test_rotation_estimator_options_default_init(estimators):
    options = estimators.RotationEstimatorOptions()
    assert options.reweighting == \
        estimators.RotationAveragingReweighting.UNIFORM
    assert options.backend == estimators.RotationAveragingBackend.L1_IRLS
    assert isinstance(options.l1_irls, estimators.L1IrlsRotationAveragerOptions)
    assert isinstance(options.ceres, estimators.CeresRotationAveragerOptions)
    assert options.ceres.loss_function_type == \
        estimators.CeresLossFunctionType.CAUCHY


def test_rotation_estimator_options_reweighting_readwrite(estimators):
    # Fork adaptation: the fork's string-to-enum helper adds an explicit
    # string constructor but no implicit str -> enum conversion on assignment.
    options = estimators.RotationEstimatorOptions()
    options.reweighting = \
        estimators.RotationAveragingReweighting.INLIER_MATCH_COUNT
    assert options.reweighting == \
        estimators.RotationAveragingReweighting.INLIER_MATCH_COUNT
    options.backend = estimators.RotationAveragingBackend("CERES")
    assert options.backend == estimators.RotationAveragingBackend.CERES
    options.ceres.loss_function_type = estimators.CeresLossFunctionType.HUBER
    assert options.ceres.loss_function_type == \
        estimators.CeresLossFunctionType.HUBER


def test_rotation_estimator_options_l1_irls_readwrite(estimators):
    options = estimators.RotationEstimatorOptions()
    assert options.l1_irls.max_num_l1_iterations == 5
    assert options.l1_irls.weight_type == \
        estimators.RotationWeightType.GEMAN_MCCLURE
    options.l1_irls.irls_loss_parameter_sigma = 2.0
    options.l1_irls.weight_type = estimators.RotationWeightType("HALF_NORM")
    assert options.l1_irls.irls_loss_parameter_sigma == 2.0
    assert options.l1_irls.weight_type == \
        estimators.RotationWeightType.HALF_NORM


def test_rotation_estimator_options_ceres_readwrite(estimators):
    options = estimators.RotationEstimatorOptions()
    assert options.ceres.max_num_warm_start_iterations == 5
    assert options.ceres.loss_function_scale == 0.5
    options.ceres.loss_function_scale = 0.1
    options.ceres.max_num_warm_start_iterations = 0
    assert options.ceres.loss_function_scale == 0.1
    assert options.ceres.max_num_warm_start_iterations == 0


def test_ceres_rotation_averager_adds_residual_and_solves(
        estimators, scene, geometry):
    # Fork adaptation: the embedded pyceres surface has no LossFunction or
    # Problem binding, so residuals added from Python use the trivial loss and
    # `problem` is the opaque CeresProblem handle exposing block counts.
    dataset_options = scene.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 2
    dataset_options.num_points3D = 10
    reconstruction = scene.Reconstruction()
    scene.synthesize_dataset(dataset_options, reconstruction, None)
    image_id1, image_id2 = sorted(reconstruction.images)

    pose_graph = scene.PoseGraph()
    edge = scene.PoseGraphEdge(cam2_from_cam1=geometry.Rigid3d())
    edge.num_matches = 10
    pose_graph.add_edge(image_id1, image_id2, edge)
    options = estimators.RotationEstimatorOptions()
    options.reweighting = \
        estimators.RotationAveragingReweighting.INLIER_MATCH_COUNT
    options.ceres.solver_options.num_threads = 1
    averager = estimators.create_default_ceres_rotation_averager(
        options, pose_graph, reconstruction)
    averager.solver_options.num_threads = 2
    assert averager.solver_options.num_threads == 2
    assert averager.problem.num_residual_blocks() == 1
    for _ in range(100):
        averager.add_relative_rotation_residual(image_id1, image_id2,
                                                geometry.Rotation3d())
    assert averager.problem.num_residual_blocks() == 101
    # keep_alive<0, 3>: the averager keeps the reconstruction alive.
    del reconstruction
    gc.collect()
    summary = averager.solve()
    assert summary.IsSolutionUsable()
    assert summary.num_threads_given == 2


def test_rotation_estimator_ceres_backend_recovers_rotations(
        estimators, scene, geometry):
    dataset_options = scene.SyntheticDatasetOptions()
    dataset_options.num_rigs = 1
    dataset_options.num_cameras_per_rig = 1
    dataset_options.num_frames_per_rig = 4
    dataset_options.num_points3D = 10
    truth = scene.Reconstruction()
    scene.synthesize_dataset(dataset_options, truth, None)
    image_ids = sorted(truth.images)

    def rotation_matrix(rec, image_id):
        w, x, y, z = rec.images[image_id].cam_from_world.rotation
        return np.asarray(geometry.Rotation3d([x, y, z, w]).matrix())

    pose_graph = scene.PoseGraph()
    for i, id1 in enumerate(image_ids):
        for id2 in image_ids[i + 1:]:
            r = rotation_matrix(truth, id2) @ rotation_matrix(truth, id1).T
            x, y, z, w = geometry.Rotation3d(r).quat
            edge = scene.PoseGraphEdge(cam2_from_cam1=geometry.Rigid3d(
                rotation_wxyz=[w, x, y, z], translation=[0, 0, 0]))
            edge.num_matches = 10
            pose_graph.add_edge(id1, id2, edge)

    options = estimators.RotationEstimatorOptions()
    options.backend = estimators.RotationAveragingBackend.CERES
    reconstruction = scene.Reconstruction(truth)
    for frame_id in list(reconstruction.reg_frame_ids):
        reconstruction.deregister_frame(frame_id)
    assert estimators.estimate_rotations(options, pose_graph, [], image_ids,
                                         reconstruction)
    for id1 in image_ids:
        for id2 in image_ids:
            est = rotation_matrix(reconstruction, id2) @ \
                rotation_matrix(reconstruction, id1).T
            ref = rotation_matrix(truth, id2) @ rotation_matrix(truth, id1).T
            np.testing.assert_allclose(est, ref, atol=1e-6)


def test_gravity_refiner_options_default_init(estimators):
    assert estimators.GravityRefinerOptions() is not None


def test_gravity_refiner_options_max_outlier_ratio_readwrite(estimators):
    options = estimators.GravityRefinerOptions()
    assert isinstance(options.max_outlier_ratio, float)
    options.max_outlier_ratio = 0.5
    assert options.max_outlier_ratio == 0.5


def test_estimate_rotations_callable(estimators):
    assert callable(estimators.estimate_rotations)
