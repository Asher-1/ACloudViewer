# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import matplotlib.pyplot as plt


def pointcloud_generator():
    yield "sphere", cv3d.geometry.ccMesh.create_sphere().\
        sample_points_uniformly(int(1e4)), 0.4

    mesh = cv3d.geometry.ccMesh.create_torus()
    # mesh.scale(5, center=mesh.get_geometry_center())
    mesh.scale(5)
    mesh += cv3d.geometry.ccMesh.create_torus()
    yield "torus", mesh.sample_points_uniformly(int(1e4)), 0.75

    d = 4
    mesh = cv3d.geometry.ccMesh.create_tetrahedron().translate((-d, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_octahedron().translate((0, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_icosahedron().translate((d, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_torus().translate((-d, -d, 0))
    mesh += cv3d.geometry.ccMesh.create_mobius(twists=1).translate((0, -d, 0))
    mesh += cv3d.geometry.ccMesh.create_mobius(twists=2).translate((d, -d, 0))
    yield "shapes", mesh.sample_points_uniformly(int(1e5)), 0.5

    ply_data = cv3d.data.PLYPointCloud()
    pcd = cv3d.io.read_point_cloud(ply_data.path)
    # Flip it, otherwise the pointcloud will be upside down.
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    yield "fragment", pcd, 0.02


if __name__ == "__main__":

    for pcl_name, pcl, eps in pointcloud_generator():
        with cv3d.utility.VerbosityContextManager(
                cv3d.utility.VerbosityLevel.Debug) as cm:
            labels = np.array(
                pcl.cluster_dbscan(eps=eps, min_points=10, print_progress=True))

        max_label = labels.max()
        print("%s has %d clusters" % (pcl_name, max_label + 1))
        colors = plt.get_cmap("tab20")(labels /
                                       (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcl.set_colors(cv3d.utility.Vector3dVector(colors[:, :3]))
        cv3d.visualization.draw([pcl])
