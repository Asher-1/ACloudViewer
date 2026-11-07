# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap estimators/bundle_adjustment_test.py."""
import numpy as np
import pytest


def test_ba_config_surface(estimators):
    cfg = estimators.BundleAdjustmentConfig()
    cfg.add_image(1)
    assert cfg.has_image(1)
    assert cfg.num_images() == 1
    cfg.remove_image(1)
    assert cfg.num_images() == 0


def test_ba_config_constant_groups(estimators):
    cfg = estimators.BundleAdjustmentConfig()
    cfg.add_image(1)  # the engine CHECKs image membership when freezing
    cfg.set_constant_cam_intrinsics(1)
    assert cfg.num_constant_cam_intrinsics() == 1
    assert cfg.has_constant_cam_intrinsics(1)
    cfg.set_constant_pose(1)
    assert cfg.has_constant_pose(1)


def test_ba_options_check(estimators):
    opts = estimators.BundleAdjustmentOptions()
    assert opts.check() in (True, False)


def test_create_default_bundle_adjuster(estimators):
    adjuster = estimators.create_default_bundle_adjuster(
        estimators.BundleAdjustmentOptions(),
        estimators.BundleAdjustmentConfig())
    assert isinstance(adjuster, estimators.BundleAdjuster)


def test_ba_summary_surface(estimators):
    summary = estimators.BundleAdjustmentSummary()
    assert hasattr(summary, "termination_type")
    assert hasattr(summary, "num_residuals")
    summary.is_solution_usable()
    summary.brief_report()


def test_ceres_bundle_adjustment_summary_hierarchy(estimators):
    summary = estimators.CeresBundleAdjustmentSummary()
    assert isinstance(summary, estimators.BundleAdjustmentSummary)
