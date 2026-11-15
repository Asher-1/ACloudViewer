# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap scene/synthetic_test.py (Open3D style)."""


def test_synthesize_dataset_default_options(scene):
    rec = scene.Reconstruction()
    scene.synthesize_dataset(scene.SyntheticDatasetOptions(), rec, None)
    assert rec.num_images == 10
    assert rec.num_points3D == 100
    assert rec.num_cameras >= 1


def test_synthesize_noise(pipeline, scene):
    rec = scene.Reconstruction()
    scene.synthesize_dataset(scene.SyntheticDatasetOptions(), rec, None)
    scene.synthesize_noise(scene.SyntheticNoiseOptions(), rec, None)
    # Noise application must not change the topology.
    assert rec.num_images == 10


def test_synthetic_noise_options_members(scene):
    opts = scene.SyntheticNoiseOptions()
    opts.prior_position_stddev = 2.0
    assert opts.prior_position_stddev == 2.0
    opts.point2D_stddev = 0.5
    assert opts.point2D_stddev == 0.5


def test_synthetic_image_options_members(scene):
    opts = scene.SyntheticImageOptions()
    opts.feature_peak_radius = 3
    assert opts.feature_peak_radius == 3
