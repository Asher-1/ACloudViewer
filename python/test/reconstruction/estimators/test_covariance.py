"""Port of upstream pycolmap estimators/covariance_test.py (Open3D style).

The fork exposes estimate_ba_covariance as a module-level entry over the
engine covariance estimator; the synthetic problem is assembled through the
CeresBundleAdjuster.
"""
import numpy as np
import pytest


def test_ba_covariance_options_surface(estimators):
    assert hasattr(estimators, "BACovarianceParams")
    assert hasattr(estimators, "BACovarianceOptions")


def test_estimate_ba_covariance_missing_reconstruction(estimators):
    adjuster = estimators.create_default_bundle_adjuster(
        estimators.BundleAdjustmentOptions(), estimators.BundleAdjustmentConfig())
    # Estimating covariance before a solve must raise rather than crash.
    with pytest.raises(Exception):
        estimators.estimate_ba_covariance(
            estimators.BACovarianceOptions(), adjuster, None)
