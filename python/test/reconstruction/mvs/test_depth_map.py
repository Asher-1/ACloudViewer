# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap mvs/depth_map_test.py (Open3D style)."""
import numpy as np


def test_depth_map_construction(mvs):
    dm = mvs.DepthMap(4, 3, 0.5, 9.0)
    arr = dm.to_array()
    assert arr.shape == (3, 4)
    assert arr.dtype == np.float32


def test_depth_map_rescale_downsize(mvs):
    dm = mvs.DepthMap(8, 4, 0.5, 9.0)
    dm.rescale(0.5)
    dm.downsize(4, 2)
    assert dm.to_array().size > 0


def test_depth_map_view_bounds(mvs):
    dm = mvs.DepthMap(4, 3, 1.0, 10.0)
    arr = dm.to_array()
    assert float(np.asarray(arr).min()) >= 0.0
