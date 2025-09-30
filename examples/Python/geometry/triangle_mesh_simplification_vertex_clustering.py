# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d

if __name__ == "__main__":
    bunny = cv3d.data.BunnyMesh()
    mesh_in = cv3d.io.read_triangle_mesh(bunny.path)
    mesh_in.compute_vertex_normals()

    print("Before Simplification: ", mesh_in)
    cv3d.visualization.draw_geometries([mesh_in])

    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=cv3d.geometry.SimplificationContraction.Average)
    print("After Simplification with voxel size =", voxel_size, ":\n", mesh_smp)
    cv3d.visualization.draw_geometries([mesh_smp])

    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=cv3d.geometry.SimplificationContraction.Average)
    print("After Simplification with voxel size =", voxel_size, ":\n", mesh_smp)
    cv3d.visualization.draw_geometries([mesh_smp])
