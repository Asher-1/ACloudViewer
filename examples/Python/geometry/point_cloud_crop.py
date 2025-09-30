# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d

if __name__ == "__main__":
    print("Load a ply point cloud, crop it, and render it")
    sample_ply_data = cv3d.data.DemoCropPointCloud()
    pcd = cv3d.io.read_point_cloud(sample_ply_data.point_cloud_path)
    vol = cv3d.visualization.read_selection_polygon_volume(
        sample_ply_data.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)
    # Flip the pointclouds, otherwise they will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    chair.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    print("Displaying original pointcloud ...")
    cv3d.visualization.draw([pcd])
    print("Displaying cropped pointcloud")
    cv3d.visualization.draw([chair])
