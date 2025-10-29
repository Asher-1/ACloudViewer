# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np

if __name__ == "__main__":
    # Read a mesh and get its data as numpy arrays.
    knot_mesh = cv3d.data.KnotMesh()
    mesh = cv3d.io.read_triangle_mesh(knot_mesh.path)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.5, 0.1, 0.3])
    print('Vertices:')
    print(np.asarray(mesh.vertices()))
    print('Vertex Colors:')
    print(np.asarray(mesh.vertex_colors()))
    print('Vertex Normals:')
    print(np.asarray(mesh.vertex_normals()))
    print('Triangles:')
    print(np.asarray(mesh.triangles()))
    print('Triangle Normals:')
    print(np.asarray(mesh.triangle_normals()))
    print("Displaying mesh ...")
    print(mesh)
    cv3d.visualization.draw([mesh])

    # Create a mesh using numpy arrays with random colors.
    N = 5
    vertices = cv3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0.5, 0.5]]))
    triangles = cv3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1], [1, 4, 2], [2, 4, 3],
                  [3, 4, 0]]))
    mesh_np = cv3d.geometry.ccMesh(vertices, triangles)
    mesh_np.set_vertex_colors(
        cv3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3))))
    mesh_np.compute_vertex_normals()
    print(np.asarray(mesh_np.triangle_normals))
    print("Displaying mesh made using numpy ...")
    cv3d.visualization.draw_geometries([mesh_np], mesh_show_wireframe=True)
