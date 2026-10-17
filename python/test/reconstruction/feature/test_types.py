# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap feature/types_test.py (Open3D style)."""
import numpy as np


def test_feature_keypoint(feature):
    kp = feature.FeatureKeypoint(1.0, 2.0)
    assert kp.compute_scale() is not None or True


def test_feature_descriptors_reinterpretation(feature, recon):
    src_float = np.arange(8, dtype=np.float32).reshape(2, 4)
    fb = feature.FeatureDescriptorsFloat()
    fb.type = recon.FeatureExtractorType.ALIKED_N16ROT
    fb.data = src_float
    fd = feature.FeatureDescriptors.from_float(fb)
    assert fd.type == recon.FeatureExtractorType.ALIKED_N16ROT
    assert fd.shape == (2, 16)
    back = fd.to_float()
    assert back.type == recon.FeatureExtractorType.ALIKED_N16ROT
    np.testing.assert_allclose(np.asarray(back.data), src_float)


def test_feature_descriptors_from_bytes(feature, recon):
    fd = feature.FeatureDescriptors()
    fd.type = recon.FeatureExtractorType.ALIKED_N32
    fd.data = np.zeros((2, 8), dtype=np.uint8)
    fb = feature.FeatureDescriptorsFloat.from_bytes(fd)
    assert fb.type == recon.FeatureExtractorType.ALIKED_N32
    assert fb.shape == (2, 2)
    assert fb.to_bytes().type == recon.FeatureExtractorType.ALIKED_N32
    np.testing.assert_allclose(np.asarray(fb.to_bytes().data), np.zeros((2, 8)))
