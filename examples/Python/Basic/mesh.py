# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/mesh.py

import copy
import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    print("Testing mesh in cloudViewer ...")
    mesh = cv3d.io.read_triangle_mesh("../../TestData/knot.ply")
    print(mesh)
    print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
    print("")

    print("Try to render a mesh with normals (exist: " +
          str(mesh.has_vertex_normals()) + ") and colors (exist: " +
          str(mesh.has_vertex_colors()) + ")")
    cv3d.visualization.draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    cv3d.visualization.draw_geometries([mesh])

    print("We make a partial mesh of only the first half triangles.")
    mesh1 = copy.deepcopy(mesh)
    mesh1.triangles = cv3d.utility.Vector3iVector(
        np.asarray(mesh1.triangles)[:len(mesh1.triangles) // 2, :])
    mesh1.triangle_normals = cv3d.utility.Vector3dVector(
        np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals) //
                                           2, :])
    print(mesh1.triangles)
    cv3d.visualization.draw_geometries([mesh1])

    print("Painting the mesh")
    mesh1.paint_uniform_color([1, 0.706, 0])
    cv3d.visualization.draw_geometries([mesh1])
