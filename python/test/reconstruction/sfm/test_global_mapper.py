# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

"""Port of upstream pycolmap sfm/global_mapper_test.py (Open3D style).

Fork adaptation: option paths are plain strings (upstream uses fs::path)."""


def test_global_mapper_options_init(sfm):
    assert sfm.GlobalMapperOptions() is not None


def test_global_mapper_options_image_path(sfm):
    options = sfm.GlobalMapperOptions()
    from pathlib import Path
    options.image_path = "/tmp/images"
    # Fork adaptation: the option round-trips as a path object.
    assert options.image_path == Path("/tmp/images")


def test_global_mapper_options_get_rotation_averaging(sfm):
    assert sfm.GlobalMapperOptions().get_rotation_averaging() is not None


def test_global_mapper_options_get_global_positioning(sfm):
    assert sfm.GlobalMapperOptions().get_global_positioning() is not None


def test_global_mapper_options_get_bundle_adjustment(sfm):
    assert sfm.GlobalMapperOptions().get_bundle_adjustment() is not None
