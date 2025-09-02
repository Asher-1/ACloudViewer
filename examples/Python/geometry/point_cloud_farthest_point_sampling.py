# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d

if __name__ == "__main__":
    # Load bunny data.
    bunny = cv3d.data.BunnyMesh()
    pcd = cv3d.io.read_point_cloud(bunny.path)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    # Get 1000 samples from original point cloud and paint to green.
    pcd_down = pcd.farthest_point_down_sample(1000)
    pcd_down.paint_uniform_color([0, 1, 0])
    cv3d.visualization.draw_geometries([pcd, pcd_down])
