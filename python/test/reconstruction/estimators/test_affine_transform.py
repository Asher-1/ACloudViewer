"""Port of upstream pycolmap estimators/affine_transform_test.py (Open3D
style). Fork adaptation: vector inputs use the utility.Vector2dVector
container."""
import numpy as np


def test_estimate_affine2d_with_points(estimators, utility):
    source = utility.Vector2dVector(
        [np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])])
    target = utility.Vector2dVector(
        [np.array([1.0, 1.0]), np.array([2.0, 1.0]), np.array([1.0, 2.0])])
    result = estimators.estimate_affine2d(source, target)
    assert result is not None
    result_array = np.array(result)
    assert result_array.shape == (2, 3)


def test_estimate_affine2d_robust_is_callable(estimators):
    assert callable(estimators.estimate_affine2d_robust)
