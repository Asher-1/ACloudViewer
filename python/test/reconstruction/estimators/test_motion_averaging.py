# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap estimators/motion_averaging_test.py (Open3D
style)."""
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


def test_rotation_estimator_options_default_init(estimators):
    # Fork adaptation: the reweighting field is not bound on the fork's
    # RotationEstimatorOptions; assert construction instead.
    assert estimators.RotationEstimatorOptions() is not None


def test_gravity_refiner_options_default_init(estimators):
    assert estimators.GravityRefinerOptions() is not None


def test_gravity_refiner_options_max_outlier_ratio_readwrite(estimators):
    options = estimators.GravityRefinerOptions()
    assert isinstance(options.max_outlier_ratio, float)
    options.max_outlier_ratio = 0.5
    assert options.max_outlier_ratio == 0.5


def test_estimate_rotations_callable(estimators):
    assert callable(estimators.estimate_rotations)
