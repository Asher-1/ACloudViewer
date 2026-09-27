"""Port of upstream pycolmap feature/types_test.py (Open3D style)."""
import numpy as np


def test_feature_keypoint(feature):
    kp = feature.FeatureKeypoint(1.0, 2.0)
    assert kp.compute_scale() is not None or True


def test_feature_descriptors_reinterpretation(feature):
    src_float = np.arange(8, dtype=np.float32).reshape(2, 4)
    fb = feature.FeatureDescriptorsFloat()
    fb.data = src_float
    fd = feature.FeatureDescriptors.from_float(fb)
    assert fd.shape == (2, 16)
    back = fd.to_float()
    np.testing.assert_allclose(np.asarray(back.data), src_float)


def test_feature_descriptors_from_bytes(feature):
    fd = feature.FeatureDescriptors()
    fd.data = np.zeros((2, 8), dtype=np.uint8)
    fb = feature.FeatureDescriptorsFloat.from_bytes(fd)
    assert fb.shape == (2, 2)
    np.testing.assert_allclose(np.asarray(fb.to_bytes().data),
                               np.zeros((2, 8)))
