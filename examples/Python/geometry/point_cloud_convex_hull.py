# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d


def mesh_generator():
    yield cv3d.geometry.ccMesh.create_box()
    yield cv3d.geometry.ccMesh.create_sphere()

    knot = cv3d.data.KnotMesh()
    mesh_knot = cv3d.io.read_triangle_mesh(knot.path)
    mesh_knot.compute_vertex_normals()
    yield mesh_knot

    bunny = cv3d.data.BunnyMesh()
    mesh_bunny = cv3d.io.read_triangle_mesh(bunny.path)
    mesh_bunny.compute_vertex_normals()
    yield mesh_bunny

    armadillo = cv3d.data.ArmadilloMesh()
    mesh_armadillo = cv3d.io.read_triangle_mesh(armadillo.path)
    mesh_armadillo.compute_vertex_normals()
    yield mesh_armadillo


if __name__ == "__main__":
    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        hull, _ = mesh.compute_convex_hull()
        hull_ls = cv3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        cv3d.visualization.draw_geometries([mesh, hull_ls])

        pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
        hull, _ = pcl.compute_convex_hull()
        hull_ls = cv3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        cv3d.visualization.draw_geometries([pcl, hull_ls])
