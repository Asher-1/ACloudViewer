# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap pipeline/sfm_test.py (Open3D style).

Fork adaptation: the options classes resolve on recon.pipeline/recon.sfm and
the fork engine entry names are used where they differ from upstream."""
import pytest


def test_view_graph_calibration_options_init(pipeline):
    assert pipeline.ViewGraphCalibrationOptions() is not None


def test_global_mapper_options_init(sfm):
    assert sfm.GlobalMapperOptions() is not None


def test_global_pipeline_options_init(sfm):
    assert sfm.GlobalPipelineOptions() is not None


def test_global_pipeline_options_min_num_matches(sfm):
    options = sfm.GlobalPipelineOptions()
    options.min_num_matches = 20
    assert options.min_num_matches == 20


@pytest.mark.parametrize(
    "name",
    [
        "bundle_adjustment",
        "triangulate_points",
        "hierarchical_mapper",
    ],
)
def test_public_api_callable(sfm, name):
    assert callable(getattr(sfm, name))


def test_calibrate_view_graph_callable(pipeline):
    assert callable(pipeline.calibrate_view_graph)
