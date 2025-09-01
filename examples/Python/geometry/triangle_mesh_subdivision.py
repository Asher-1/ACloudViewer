# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/Python/Basic/mesh_subdivision.py

import numpy as np
import cloudViewer as cv3d


def mesh_generator():
    knot_mesh = cv3d.data.KnotMesh()
    mesh = cv3d.io.read_triangle_mesh(knot_mesh.path)
    mesh.compute_vertex_normals()
    yield mesh
    yield cv3d.geometry.ccMesh.create_plane()
    yield cv3d.geometry.ccMesh.create_tetrahedron()
    yield cv3d.geometry.ccMesh.create_box()
    yield cv3d.geometry.ccMesh.create_octahedron()
    yield cv3d.geometry.ccMesh.create_icosahedron()
    yield cv3d.geometry.ccMesh.create_sphere()
    yield cv3d.geometry.ccMesh.create_cone()
    yield cv3d.geometry.ccMesh.create_cylinder()


if __name__ == "__main__":
    np.random.seed(42)

    number_of_iterations = 3

    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        n_verts = np.asarray(mesh.get_vertices()).shape[0]
        colors = np.random.uniform(0, 1, size=(n_verts, 3))
        mesh.set_vertex_colors(cv3d.utility.Vector3dVector(colors))

        print("original mesh has %d triangles and %d vertices" %
              (np.asarray(mesh.get_triangles()).shape[0],
               np.asarray(mesh.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh])

        mesh_up = mesh.subdivide_midpoint(
            number_of_iterations=number_of_iterations)
        print("midpoint upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.get_triangles()).shape[0],
               np.asarray(mesh_up.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh_up])

        mesh_up = mesh.subdivide_loop(number_of_iterations=number_of_iterations)
        print("loop upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.get_triangles()).shape[0],
               np.asarray(mesh_up.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh_up])
