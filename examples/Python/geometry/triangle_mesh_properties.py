# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import os
import sys

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

import cloudViewer_example as cv3dex


def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    geoms = [mesh]
    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        geoms.append(cv3dex.edges_to_lineset(mesh, edges, (1, 0, 0)))
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        geoms.append(cv3dex.edges_to_lineset(mesh, edges, (0, 1, 0)))
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        pcl = cv3d.geometry.ccPointCloud(
            points=cv3d.utility.Vector3dVector(np.asarray(mesh.get_vertices())[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        geoms.append(pcl)
    if self_intersecting:
        intersecting_triangles = np.asarray(
            mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles = np.unique(intersecting_triangles)
        print("  # visualize self-intersecting triangles")
        triangles = np.asarray(mesh.get_triangles())[intersecting_triangles]
        edges = [
            np.vstack((triangles[:, i], triangles[:, j]))
            for i, j in [(0, 1), (1, 2), (2, 0)]
        ]
        edges = np.hstack(edges).T
        edges = cv3d.utility.Vector2iVector(edges)
        geoms.append(cv3dex.edges_to_lineset(mesh, edges, (1, 0, 1)))
    cv3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)


if __name__ == "__main__":
    knot_mesh = cv3d.data.KnotMesh()
    mesh = cv3d.io.read_triangle_mesh(knot_mesh.path)
    check_properties('KnotMesh', mesh)
    check_properties('Mobius',
                     cv3d.geometry.ccMesh.create_moebius(twists=1))
    check_properties("non-manifold edge", cv3dex.get_non_manifold_edge_mesh())
    check_properties("non-manifold vertex",
                     cv3dex.get_non_manifold_vertex_mesh())
    check_properties("open box", cv3dex.get_open_box_mesh())
    check_properties("intersecting_boxes", cv3dex.get_intersecting_boxes_mesh())
