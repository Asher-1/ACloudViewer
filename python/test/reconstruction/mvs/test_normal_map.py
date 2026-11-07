# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Port of upstream pycolmap mvs/normal_map_test.py (Open3D style)."""
import numpy as np


def test_normal_map_construction(mvs):
    nm = mvs.NormalMap(4, 3)
    arr = nm.to_array()
    assert arr.shape == (3, 4, 3)


def test_normal_map_to_bitmap(mvs):
    nm = mvs.NormalMap(4, 3)
    assert nm.to_bitmap() is not None
