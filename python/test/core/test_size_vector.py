# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import pytest

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from cloudViewer_test import list_devices


def test_size_vector():
    # List
    sv = cv3d.core.SizeVector([-1, 2, 3])
    assert "{}".format(sv) == "SizeVector[-1, 2, 3]"

    # Tuple
    sv = cv3d.core.SizeVector((-1, 2, 3))
    assert "{}".format(sv) == "SizeVector[-1, 2, 3]"

    # Numpy 1D array
    sv = cv3d.core.SizeVector(np.array([-1, 2, 3]))
    assert "{}".format(sv) == "SizeVector[-1, 2, 3]"

    # Empty
    sv = cv3d.core.SizeVector()
    assert "{}".format(sv) == "SizeVector[]"
    sv = cv3d.core.SizeVector([])
    assert "{}".format(sv) == "SizeVector[]"
    sv = cv3d.core.SizeVector(())
    assert "{}".format(sv) == "SizeVector[]"
    sv = cv3d.core.SizeVector(np.array([]))
    assert "{}".format(sv) == "SizeVector[]"

    # 1-dimensional SizeVector
    assert cv3d.core.SizeVector(3) == (3,)
    assert cv3d.core.SizeVector((3)) == (3,)
    assert cv3d.core.SizeVector((3,)) == (3,)
    assert cv3d.core.SizeVector([3]) == (3,)
    assert cv3d.core.SizeVector([
        3,
    ]) == (3,)

    # Not integer: thorws exception
    with pytest.raises(Exception):
        sv = cv3d.core.SizeVector([1.9, 2, 3])

    with pytest.raises(Exception):
        sv = cv3d.core.SizeVector([-1.5, 2, 3])

    # 2D list exception
    with pytest.raises(Exception):
        sv = cv3d.core.SizeVector([[1, 2], [3, 4]])

    # 2D Numpy array exception
    with pytest.raises(Exception):
        sv = cv3d.core.SizeVector(np.array([[1, 2], [3, 4]]))

    # Garbage input
    with pytest.raises(Exception):
        sv = cv3d.core.SizeVector(["foo", "bar"])


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
def test_implicit_conversion(device):
    # Reshape
    t = cv3d.core.Tensor.ones((3, 4), device=device)
    assert t.reshape(cv3d.core.SizeVector((4, 3))).shape == (4, 3)
    assert t.reshape(cv3d.core.SizeVector([4, 3])).shape == (4, 3)
    assert t.reshape((4, 3)).shape == (4, 3)
    assert t.reshape([4, 3]).shape == (4, 3)
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.reshape((4, 3.0))
    with pytest.raises(TypeError, match="incompatible function arguments"):
        t.reshape((4.0, 3.0))
    with pytest.raises(RuntimeError, match="Invalid shape dimension"):
        t.reshape((4, -3))

    # 0-dimensional
    assert cv3d.core.Tensor.ones((), device=device).shape == ()
    assert cv3d.core.Tensor.ones([], device=device).shape == ()

    # 1-dimensional
    assert cv3d.core.Tensor.ones(3, device=device).shape == (3,)
    assert cv3d.core.Tensor.ones((3), device=device).shape == (3,)
    assert cv3d.core.Tensor.ones((3,), device=device).shape == (3,)
    assert cv3d.core.Tensor.ones([3], device=device).shape == (3,)
    assert cv3d.core.Tensor.ones([
        3,
    ], device=device).shape == (3,)

    # Tensor creation
    assert cv3d.core.Tensor.empty((3, 4), device=device).shape == (3, 4)
    assert cv3d.core.Tensor.ones((3, 4), device=device).shape == (3, 4)
    assert cv3d.core.Tensor.zeros((3, 4), device=device).shape == (3, 4)
    assert cv3d.core.Tensor.full((3, 4), 10, device=device).shape == (3, 4)

    # Reduction
    t = cv3d.core.Tensor.ones((3, 4, 5), device=device)
    assert t.sum(cv3d.core.SizeVector([0, 2])).shape == (4,)
    assert t.sum(cv3d.core.SizeVector([0, 2]), keepdim=True).shape == (1, 4, 1)
    assert t.sum((0, 2)).shape == (4,)
    assert t.sum([0, 2]).shape == (4,)
    assert t.sum((0, 2), keepdim=True).shape == (1, 4, 1)
    assert t.sum([0, 2], keepdim=True).shape == (1, 4, 1)
