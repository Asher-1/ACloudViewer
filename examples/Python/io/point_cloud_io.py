# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d

if __name__ == "__main__":
    pcd_data = cv3d.data.PCDPointCloud()
    print(
        f"Reading pointcloud from file: fragment.pcd stored at {pcd_data.path}")
    pcd = cv3d.io.read_point_cloud(pcd_data.path)
    print(pcd)
    print("Saving pointcloud to file: copy_of_fragment.pcd")
    cv3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
