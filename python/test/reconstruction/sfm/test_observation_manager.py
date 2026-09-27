# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap sfm/observation_manager_test.py (Open3D
style)."""
import pytest


def test_image_pair_stat_init(sfm):
    assert sfm.ImagePairStat() is not None


def test_image_pair_stat_num_tri_corrs(sfm):
    stat = sfm.ImagePairStat()
    stat.num_tri_corrs = 10
    assert stat.num_tri_corrs == 10


def test_image_pair_stat_num_total_corrs(sfm):
    stat = sfm.ImagePairStat()
    stat.num_total_corrs = 20
    assert stat.num_total_corrs == 20


def test_reprojection_error_type_pixel(scene):
    assert scene.ReprojectionErrorType.PIXEL is not None


def test_reprojection_error_type_normalized(scene):
    assert scene.ReprojectionErrorType.NORMALIZED is not None


def test_reprojection_error_type_angular(scene):
    assert scene.ReprojectionErrorType.ANGULAR is not None


def test_observation_manager_class_exists(sfm):
    assert hasattr(sfm, "ObservationManager")
