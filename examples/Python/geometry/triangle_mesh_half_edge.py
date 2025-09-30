# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import cloudViewer as cv3d


if __name__ == "__main__":
    # Initialize a HalfEdgeTriangleMesh from TriangleMesh
    mesh = cv3d.geometry.ccMesh.create_sphere()
    bbox = cv3d.geometry.ccBBox()
    bbox.set_min_bound([-1, -1, -1])
    bbox.set_max_bound([1, 0.6, 1])
    bbox.set_validity(True)
    mesh = mesh.crop(bbox)
    het_mesh = cv3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
    cv3d.visualization.draw_geometries([het_mesh], mesh_show_back_face=True)

    # Colorize boundary vertices to red
    vertex_colors = 0.75 * np.ones((len(het_mesh.vertices), 3))
    for boundary in het_mesh.get_boundaries():
        for vertex_id in boundary:
            vertex_colors[vertex_id] = [1, 0, 0]
    het_mesh.vertex_colors = cv3d.utility.Vector3dVector(vertex_colors)
    cv3d.visualization.draw_geometries([het_mesh], mesh_show_back_face=True)
