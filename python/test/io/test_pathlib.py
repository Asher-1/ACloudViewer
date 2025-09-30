# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from pathlib import Path

import cloudViewer as cv3d


def test_pathlib_support():
    pcd_pointcloud = cv3d.data.PCDPointCloud()
    assert isinstance(pcd_pointcloud.path, str)

    pcd = cv3d.io.read_point_cloud(pcd_pointcloud.path)
    assert pcd.has_points()

    pcd = cv3d.io.read_point_cloud(Path(pcd_pointcloud.path))
    assert pcd.has_points()
