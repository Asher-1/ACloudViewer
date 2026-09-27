"""Port of upstream pycolmap pipeline/extract_features_test.py (Open3D
style). The upstream pybind surface exposes ``pycolmap.extract_features``
(plural); the fork's pairing-driven engine surface exposes
``recon.feature.extract_feature`` (singular), so the callable probe targets
the fork name."""
import pytest


def test_extract_feature_callable(feature):
    assert callable(feature.extract_feature)


def test_extract_feature_signature_rejects_missing_args(feature):
    # The engine entry point requires a database and image inputs; calling
    # it without arguments must raise, not crash.
    with pytest.raises(TypeError):
        feature.extract_feature()
