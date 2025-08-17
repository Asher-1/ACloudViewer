# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import cloudViewer.core as o3c
import numpy as np
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from cloudViewer_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_tensorlistmap(device):
    dtype = o3c.Dtype.Float32

    # Constructor.
    tlm = cv3d.t.geometry.TensorListMap("points")

    # Get primary key().
    assert tlm.get_primary_key() == "points"

    # Map member access, assignment and "contains" check. This should be the
    # preferrred way to construct a TensorListMap with values in python.
    points = o3c.TensorList(o3c.SizeVector([3]), dtype, device)
    colors = o3c.TensorList(o3c.SizeVector([3]), dtype, device)
    tlm = cv3d.t.geometry.TensorListMap("points")
    assert "points" not in tlm
    tlm["points"] = points
    assert "points" in tlm
    assert "colors" not in tlm
    tlm["colors"] = colors
    assert "colors" in tlm

    # Constructor with tl values.
    tlm = cv3d.t.geometry.TensorListMap("points", {
        "points": points,
        "colors": colors
    })

    # Syncronized pushback.
    one_point = o3c.Tensor.ones((3,), dtype, device)
    one_color = o3c.Tensor.ones((3,), dtype, device)
    tlm.synchronized_push_back({"points": one_point, "colors": one_color})
    tlm["points"].push_back(one_point)
    assert not tlm.is_size_synchronized()
    with pytest.raises(RuntimeError):
        tlm.assert_size_synchronized()
    with pytest.raises(RuntimeError):
        tlm.synchronized_push_back({"points": one_point, "colors": one_color})
