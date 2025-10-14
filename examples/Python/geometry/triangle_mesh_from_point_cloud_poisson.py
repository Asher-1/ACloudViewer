# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    eagle = cv3d.data.EaglePointCloud()
    pcd = cv3d.io.read_point_cloud(eagle.path)
    R = pcd.get_rotation_matrix_from_xyz((np.pi, -np.pi / 4, 0))
    pcd.rotate(R, center=(0, 0, 0))
    print('Displaying input pointcloud ...')
    cv3d.visualization.draw([pcd])

    print('Running Poisson surface reconstruction ...')
    mesh, densities = cv3d.geometry.ccMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    print('Displaying reconstructed mesh ...')
    cv3d.visualization.draw([mesh])

    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = cv3d.geometry.ccMesh()
    density_mesh.create_internal_cloud()
    density_mesh.set_vertices(mesh.get_vertices())
    density_mesh.set_triangles(mesh.get_triangles())
    density_mesh.set_vertex_colors(cv3d.utility.Vector3dVector(density_colors))
    cv3d.visualization.draw_geometries([density_mesh])

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    cv3d.visualization.draw_geometries([mesh])
