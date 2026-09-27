# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap feature/matching_test.py (Open3D style).

Adaptations to the fork pybind surface:
- The new-abstraction types (FeatureMatchingOptions / FeatureMatcher /
  FeatureMatcherType) are registered on the reconstruction root module, so
  they are reached through the ``recon`` fixture while SiftMatchingOptions
  stays on the ``feature`` submodule.
- ``FeatureMatcher.create`` takes the matching ``options`` (the fork has no
  top-level device argument; engine placement follows the options).
- The upstream ALIKED backends resolve through the AICore ggml task family;
  until that task lands the factory throws at create time, mirroring
  upstream's ONNX-disabled behavior.
"""
import pytest


def test_feature_matcher_type_enum(recon):
    assert recon.FeatureMatcherType.UNDEFINED is not None
    assert recon.FeatureMatcherType.SIFT_BRUTEFORCE is not None


def test_sift_matching_options_default_init(feature):
    options = feature.SiftMatchingOptions()
    assert options is not None


def test_sift_matching_options_check(feature):
    options = feature.SiftMatchingOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_matching_options_default_init(recon):
    options = recon.FeatureMatchingOptions()
    assert options is not None


def test_feature_matching_options_type_readwrite(recon):
    options = recon.FeatureMatchingOptions()
    options.type = recon.FeatureMatcherType.SIFT_BRUTEFORCE
    assert options.type == recon.FeatureMatcherType.SIFT_BRUTEFORCE


def test_feature_matching_options_num_threads_readwrite(recon):
    options = recon.FeatureMatchingOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_feature_matching_options_use_gpu_readwrite(recon):
    options = recon.FeatureMatchingOptions()
    options.use_gpu = False
    assert options.use_gpu is False


def test_feature_matching_options_max_num_matches_readwrite(recon):
    options = recon.FeatureMatchingOptions()
    options.max_num_matches = 16384
    assert options.max_num_matches == 16384


def test_feature_matching_options_guided_matching_readwrite(recon):
    options = recon.FeatureMatchingOptions()
    options.guided_matching = True
    assert options.guided_matching is True
    options.guided_matching = False
    assert options.guided_matching is False


def test_feature_matching_options_skip_geometric_verification_readwrite(recon):
    options = recon.FeatureMatchingOptions()
    options.skip_geometric_verification = True
    assert options.skip_geometric_verification is True
    options.skip_geometric_verification = False
    assert options.skip_geometric_verification is False


def test_feature_matching_options_check(recon):
    options = recon.FeatureMatchingOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_matching_options_check_loma_resolves_defaults(recon):
    # Default-constructed LoMa options carry empty model paths; Check()
    # resolves the catalog defaults (no download) instead of failing.
    options = recon.FeatureMatchingOptions(recon.FeatureMatcherType.LOMA_B)
    assert options.check()


def test_feature_matcher_create(recon):
    matcher = recon.FeatureMatcher.create(recon.FeatureMatchingOptions())
    assert matcher is not None


def test_feature_matcher_create_aliked_registered(recon):
    # Batch A engine alignment: the ALIKED backends now resolve through the
    # AICore ggml task family, so the factory constructs successfully
    # (upstream ONNX-enabled parity); the brute-force path needs no model.
    options = recon.FeatureMatchingOptions(
        recon.FeatureMatcherType.ALIKED_BRUTEFORCE)
    assert options.check()
    matcher = recon.FeatureMatcher.create(options)
    assert matcher is not None
