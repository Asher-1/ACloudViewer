# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/Python/Basic/mesh_filter.py

import numpy as np
import cloudViewer as cv3d


def test_mesh(noise=0):
    knot = cv3d.data.KnotMesh()
    mesh = cv3d.io.read_triangle_mesh(knot.path)
    mesh.compute_vertex_normals()
    if noise > 0:
        vertices = np.asarray(mesh.get_vertices())
        vertices += np.random.uniform(0, noise, size=vertices.shape)
        mesh.set_vertices(cv3d.utility.Vector3dVector(vertices))
    mesh.compute_vertex_normals()
    return mesh


if __name__ == '__main__':
    in_mesh = test_mesh()
    cv3d.visualization.draw_geometries([in_mesh])

    mesh = in_mesh.filter_sharpen(number_of_iterations=1, strength=1)
    cv3d.visualization.draw_geometries([mesh])

    in_mesh = test_mesh(noise=5)
    cv3d.visualization.draw_geometries([in_mesh])

    mesh = in_mesh.filter_smooth_simple(number_of_iterations=1)
    cv3d.visualization.draw_geometries([mesh])

    cv3d.visualization.draw_geometries([in_mesh])
    mesh = in_mesh.filter_smooth_laplacian(number_of_iterations=100)
    cv3d.visualization.draw_geometries([mesh])

    cv3d.visualization.draw_geometries([in_mesh])
    mesh = in_mesh.filter_smooth_taubin(number_of_iterations=100)
    cv3d.visualization.draw_geometries([mesh])
