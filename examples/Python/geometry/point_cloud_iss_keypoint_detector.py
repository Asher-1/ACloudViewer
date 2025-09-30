# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import time

if __name__ == "__main__":
    # Compute ISS Keypoints on armadillo pointcloud.
    armadillo_data = cv3d.data.ArmadilloMesh()
    mesh = cv3d.io.read_triangle_mesh(armadillo_data.path)
    pcd = cv3d.geometry.ccPointCloud()
    pcd.set_points(cv3d.utility.Vector3dVector(mesh.vertices()))

    tic = time.time()
    keypoints = cv3d.geometry.keypoint.compute_iss_keypoints(pcd)
    toc = 1000 * (time.time() - tic)
    print("ISS Computation took {:.0f} [ms]".format(toc))

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    keypoints.paint_uniform_color([1.0, 0.0, 0.0])
    cv3d.visualization.draw([keypoints, mesh], point_size=5)
