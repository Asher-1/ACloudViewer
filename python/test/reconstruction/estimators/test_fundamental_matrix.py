"""Port of upstream pycolmap estimators/fundamental_matrix_test.py (Open3D
style)."""


def test_estimate_fundamental_matrix_is_callable(estimators):
    assert callable(estimators.estimate_fundamental_matrix)
