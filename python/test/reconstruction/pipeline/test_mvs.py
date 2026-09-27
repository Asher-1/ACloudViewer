# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap pipeline/mvs_test.py (Open3D style).

Fork adaptation: the MVS surface lives on recon.mvs; the engine entry points
are stereo_patch_match / stereo_fuse."""
import pytest


def test_patch_match_options_init(mvs):
    assert mvs.PatchMatchOptions() is not None


def test_stereo_fusion_options_init(mvs):
    assert mvs.StereoFusionOptions() is not None


@pytest.mark.parametrize(
    "name",
    [
        "stereo_patch_match",
        "stereo_fuse",
    ],
)
def test_public_api_callable(mvs, name):
    assert callable(getattr(mvs, name))
