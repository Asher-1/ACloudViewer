# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import cloudViewer as cv3d


def average_filtering():
    # Create noisy mesh.
    knot_mesh = cv3d.data.KnotMesh()
    mesh_in = cv3d.io.read_triangle_mesh(knot_mesh.path)
    vertices = np.asarray(mesh_in.vertices())
    noise = 5
    vertices += np.random.uniform(0, noise, size=vertices.shape)
    mesh_in.set_vertices(cv3d.utility.Vector3dVector(vertices))
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    cv3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of average mesh filter after 1 iteration ...")
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=1)
    mesh_out.compute_vertex_normals()
    cv3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of average mesh filter after 5 iteration ...")
    mesh_out = mesh_in.filter_smooth_simple(number_of_iterations=5)
    mesh_out.compute_vertex_normals()
    cv3d.visualization.draw_geometries([mesh_out])


def laplace_filtering():
    # Create noisy mesh.
    knot_mesh = cv3d.data.KnotMesh()
    mesh_in = cv3d.io.read_triangle_mesh(knot_mesh.path)
    vertices = np.asarray(mesh_in.vertices())
    noise = 5
    vertices += np.random.uniform(0, noise, size=vertices.shape)
    mesh_in.set_vertices(cv3d.utility.Vector3dVector(vertices))
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    cv3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of Laplace mesh filter after 10 iteration ...")
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    cv3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of Laplace mesh filter after 50 iteration ...")
    mesh_out = mesh_in.filter_smooth_laplacian(number_of_iterations=50)
    mesh_out.compute_vertex_normals()
    cv3d.visualization.draw_geometries([mesh_out])


def taubin_filtering():
    # Create noisy mesh.
    knot_mesh = cv3d.data.KnotMesh()
    mesh_in = cv3d.io.read_triangle_mesh(knot_mesh.path)
    vertices = np.asarray(mesh_in.vertices())
    noise = 5
    vertices += np.random.uniform(0, noise, size=vertices.shape)
    mesh_in.set_vertices(cv3d.utility.Vector3dVector(vertices))
    mesh_in.compute_vertex_normals()
    print("Displaying input mesh ...")
    cv3d.visualization.draw_geometries([mesh_in])

    print("Displaying output of Taubin mesh filter after 10 iteration ...")
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=10)
    mesh_out.compute_vertex_normals()
    cv3d.visualization.draw_geometries([mesh_out])

    print("Displaying output of Taubin mesh filter after 100 iteration ...")
    mesh_out = mesh_in.filter_smooth_taubin(number_of_iterations=100)
    mesh_out.compute_vertex_normals()
    cv3d.visualization.draw_geometries([mesh_out])


if __name__ == "__main__":
    average_filtering()
    laplace_filtering()
    taubin_filtering()
