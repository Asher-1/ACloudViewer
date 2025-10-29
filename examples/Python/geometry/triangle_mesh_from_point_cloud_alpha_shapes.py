# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np

if __name__ == "__main__":
    bunny = cv3d.data.BunnyMesh()
    mesh = cv3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    pcd = mesh.sample_points_poisson_disk(750)
    print("Displaying input pointcloud ...")
    cv3d.visualization.draw_geometries([pcd])
    alpha = 0.03
    print(f"alpha={alpha:.3f}")
    print('Running alpha shapes surface reconstruction ...')
    mesh = cv3d.geometry.ccMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_triangle_normals(normalized=True)
    print("Displaying reconstructed mesh ...")
    cv3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    tetra_mesh, pt_map = cv3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    print("done with tetra mesh")
    cv3d.visualization.draw_geometries([tetra_mesh])
    for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
        print("alpha={}".format(alpha))
        mesh = cv3d.geometry.ccMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        cv3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
