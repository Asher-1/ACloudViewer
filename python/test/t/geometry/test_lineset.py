# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import pytest
import pickle
import tempfile

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from cloudViewer_test import list_devices


def test_extrude_rotation():
    line = cv3d.t.geometry.LineSet([[0.7, 0, 0], [1, 0, 0]], [[0, 1]])
    ans = line.extrude_rotation(3 * 360, [0, 1, 0],
                                resolution=3 * 16,
                                translation=2)
    assert ans.vertex.positions.shape == (98, 3)
    assert ans.triangle.indices.shape == (96, 3)


def test_extrude_linear():
    lines = cv3d.t.geometry.LineSet([[1.0, 0.0, 0.0], [0, 0, 0], [0, 0, 1]],
                                    [[0, 1], [1, 2]])
    ans = lines.extrude_linear([0, 1, 0])
    assert ans.vertex.positions.shape == (6, 3)
    assert ans.triangle.indices.shape == (4, 3)


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
def test_pickle(device):
    line = cv3d.t.geometry.LineSet([[0.7, 0, 0], [1, 0, 0]],
                                   [[0, 1]]).to(device)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/lineset.pkl"
        pickle.dump(line, open(file_name, "wb"))
        line_load = pickle.load(open(file_name, "rb"))
        assert line_load.device == device
        np.testing.assert_equal(line_load.point.positions.cpu().numpy(),
                                line.point.positions.cpu().numpy())
        np.testing.assert_equal(line_load.line.indices.cpu().numpy(),
                                line.line.indices.cpu().numpy())
