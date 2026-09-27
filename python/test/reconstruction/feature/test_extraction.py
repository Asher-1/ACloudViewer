# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap feature/extraction_test.py (Open3D style).

Adaptations to the fork pybind surface:
- ``pycolmap.Normalization`` is carried as the ``NormalizationType``
  native enum nested on ``SiftExtractionOptions`` (bound on recon.feature).
- The new-abstraction types (FeatureExtractionOptions / FeatureExtractor /
  enums) are registered on the reconstruction root module, so they are
  reached through the ``recon`` fixture while the SIFT option classes stay
  on the ``feature`` submodule.
- ``FeatureExtractor.create`` takes the extraction ``options`` (the fork
  has no top-level device argument; engine placement follows the options).
"""
import pytest


def test_normalization_enum(feature):
    normalization = feature.SiftExtractionOptions.NormalizationType
    assert normalization.L1_ROOT is not None
    assert normalization.L2 is not None


def test_sift_extraction_options_default_init(feature):
    options = feature.SiftExtractionOptions()
    assert options is not None


def test_sift_extraction_options_num_octaves_readwrite(feature):
    options = feature.SiftExtractionOptions()
    original = options.num_octaves
    assert isinstance(original, int)
    options.num_octaves = 5
    assert options.num_octaves == 5


def test_sift_extraction_options_max_num_features_readwrite(feature):
    options = feature.SiftExtractionOptions()
    assert isinstance(options.max_num_features, int)
    options.max_num_features = 4096
    assert options.max_num_features == 4096


def test_sift_extraction_options_peak_threshold_readwrite(feature):
    options = feature.SiftExtractionOptions()
    assert isinstance(options.peak_threshold, float)
    options.peak_threshold = 0.01
    assert abs(options.peak_threshold - 0.01) < 1e-6


def test_sift_extraction_options_check(feature):
    options = feature.SiftExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extraction_options_default_init(recon):
    options = recon.FeatureExtractionOptions()
    assert options is not None


def test_feature_extraction_options_max_image_size_readwrite(recon):
    options = recon.FeatureExtractionOptions()
    options.max_image_size = 2048
    assert options.max_image_size == 2048


def test_feature_extraction_options_num_threads_readwrite(recon):
    options = recon.FeatureExtractionOptions()
    options.num_threads = 4
    assert options.num_threads == 4


def test_feature_extraction_options_use_gpu_readwrite(recon):
    options = recon.FeatureExtractionOptions()
    options.use_gpu = False
    assert options.use_gpu is False


def test_feature_extraction_options_sift_property(recon, feature):
    options = recon.FeatureExtractionOptions()
    sift = options.sift
    assert isinstance(sift, feature.SiftExtractionOptions)


def test_feature_extraction_options_requires_rgb(recon):
    options = recon.FeatureExtractionOptions()
    result = options.requires_rgb()
    assert isinstance(result, bool)


def test_feature_extraction_options_requires_rgb_by_type(recon):
    options = recon.FeatureExtractionOptions()
    assert not options.requires_rgb()
    options.type = recon.FeatureExtractorType.LOMA_B
    assert options.requires_rgb()


def test_feature_extraction_options_requires_opengl(recon):
    options = recon.FeatureExtractionOptions()
    options.use_gpu = False
    result = options.requires_open_gl()
    assert isinstance(result, bool)
    assert result is False


def test_feature_extraction_options_eff_max_image_size(recon):
    options = recon.FeatureExtractionOptions()
    # max_image_size defaults to -1: the type-specific default (SIFT 3200)
    # must be selected, mirroring upstream EffMaxImageSize.
    result = options.eff_max_image_size()
    assert isinstance(result, int)
    assert result > 0


def test_feature_extraction_options_check(recon):
    options = recon.FeatureExtractionOptions()
    result = options.check()
    assert isinstance(result, bool)
    assert result is True


def test_feature_extractor_create(recon):
    extractor = recon.FeatureExtractor.create(recon.FeatureExtractionOptions())
    assert extractor is not None


def test_feature_extractor_create_loma_options_resolve(recon):
    # The LoMa shell must pass options.Check() with catalog-default paths
    # (resolution only, no download); the ALIKED branch is expected to
    # throw until its AICore ggml task lands.
    options = recon.FeatureExtractionOptions(recon.FeatureExtractorType.LOMA_B)
    assert options.check()
    extractor = recon.FeatureExtractor.create(options)
    assert extractor is not None


def test_feature_extractor_create_aliked_not_registered(recon):
    options = recon.FeatureExtractionOptions(
        recon.FeatureExtractorType.ALIKED_N16ROT)
    assert options.check()
    with pytest.raises(RuntimeError):
        recon.FeatureExtractor.create(options)
