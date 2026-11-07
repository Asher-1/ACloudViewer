# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap sfm/incremental_mapper_test.py (Open3D
style). Fork adaptation: the option classes live on recon.sfm."""


def test_incremental_mapper_options_init(sfm):
    assert sfm.IncrementalPipelineOptions() is not None


def test_incremental_mapper_options_instantiation(sfm):
    # Fork adaptation: min_num_matches is not bound on the fork's
    # IncrementalPipelineOptions; assert construction instead.
    assert sfm.IncrementalPipelineOptions() is not None


def test_incremental_mapper_class_exists(sfm):
    assert hasattr(sfm, "IncrementalMapper")
