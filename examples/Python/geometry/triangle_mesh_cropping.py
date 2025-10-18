# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/Python/Basic/mesh_filter.py

import numpy as np
import cloudViewer as cv3d
import copy

if __name__ == "__main__":
    knot_mesh = cv3d.data.KnotMesh()
    mesh = cv3d.io.read_triangle_mesh(knot_mesh.path)
    mesh.compute_vertex_normals()
    print("Displaying original mesh ...")
    cv3d.visualization.draw([mesh])

    print("Displaying mesh of only the first half triangles ...")
    mesh_cropped = copy.deepcopy(mesh)
    mesh_cropped.set_triangles(
        cv3d.utility.Vector3iVector(
            np.asarray(
                mesh_cropped.triangles())[:len(mesh_cropped.triangles()) //
                                          2, :]))
    mesh_cropped.set_triangle_normals(
        cv3d.utility.Vector3dVector(
            np.asarray(mesh_cropped.triangle_normals())
            [:len(mesh_cropped.triangle_normals()) // 2, :]))
    print(mesh_cropped.triangles())
    cv3d.visualization.draw([mesh_cropped])
