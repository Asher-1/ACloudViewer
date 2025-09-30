# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d

if __name__ == "__main__":
    print("Load a ply point cloud, print it, and render it")
    sample_ply_data = cv3d.data.PLYPointCloud()
    pcd = cv3d.io.read_point_cloud(sample_ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    print(pcd)
    cv3d.visualization.draw([pcd])
    print("Paint pointcloud")
    pcd.paint_uniform_color([1, 0.706, 0])
    cv3d.visualization.draw([pcd])
