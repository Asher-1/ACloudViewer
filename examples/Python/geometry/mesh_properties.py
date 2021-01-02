# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/mesh_properties.py

import numpy as np
import cloudViewer as cv3d
import time

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../misc"))
import meshes


def mesh_generator(edge_cases=True):
    # yield "box", cv3d.geometry.ccMesh.create_box()
    # yield "sphere", cv3d.geometry.ccMesh.create_sphere()
    # yield "cone", cv3d.geometry.ccMesh.create_cone()
    # yield "torus", cv3d.geometry.ccMesh.create_torus(radial_resolution=30, tubular_resolution=20)
    yield "moebius (twists=1)", cv3d.geometry.ccMesh.create_moebius(
        twists=1)
    yield "moebius (twists=2)", cv3d.geometry.ccMesh.create_moebius(
        twists=2)
    yield "moebius (twists=3)", cv3d.geometry.ccMesh.create_moebius(
        twists=3)

    yield "knot", meshes.knot()

    if edge_cases:
        yield "non-manifold edge", meshes.non_manifold_edge()
        yield "non-manifold vertex", meshes.non_manifold_vertex()
        yield "open box", meshes.open_box()
        yield "boxes", meshes.intersecting_boxes()


def check_properties(name, mesh):
    def fmt_bool(b):
        return "yes" if b else "no"

    print(name)
    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    print("  edge_manifold:          %s" % fmt_bool(edge_manifold))
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    print("  edge_manifold_boundary: %s" % fmt_bool(edge_manifold_boundary))
    vertex_manifold = mesh.is_vertex_manifold()
    print("  vertex_manifold:        %s" % fmt_bool(vertex_manifold))
    self_intersecting = mesh.is_self_intersecting()
    print("  self_intersecting:      %s" % fmt_bool(self_intersecting))
    watertight = mesh.is_watertight()
    print("  watertight:             %s" % fmt_bool(watertight))
    orientable = mesh.is_orientable()
    print("  orientable:             %s" % fmt_bool(orientable))

    mesh.compute_vertex_normals()

    if not edge_manifold:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
        print("  # visualize non-manifold edges (allow_boundary_edges=True)")
        cv3d.visualization.draw_geometries(
            [mesh, meshes.edges_to_lineset(mesh, edges, (1, 0, 0))])
    if not edge_manifold_boundary:
        edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
        print("  # visualize non-manifold edges (allow_boundary_edges=False)")
        cv3d.visualization.draw_geometries(
            [mesh, meshes.edges_to_lineset(mesh, edges, (0, 1, 0))])
    if not vertex_manifold:
        verts = np.asarray(mesh.get_non_manifold_vertices())
        print("  # visualize non-manifold vertices")
        pcl = cv3d.geometry.ccPointCloud(
            points=cv3d.utility.Vector3dVector(np.asarray(mesh.get_vertices())[verts]))
        pcl.paint_uniform_color((0, 0, 1))
        cv3d.visualization.draw_geometries([mesh, pcl])
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
        cv3d.visualization.draw_geometries(
            [mesh, meshes.edges_to_lineset(mesh, edges, (1, 1, 0))])
    if watertight:
        print("  # visualize watertight mesh")
        cv3d.visualization.draw_geometries([mesh])

    if not edge_manifold:
        print("  # Remove non-manifold edges")
        mesh.remove_non_manifold_edges()
        print("  # Is mesh now edge-manifold: {}".format(
            fmt_bool(mesh.is_edge_manifold())))
        cv3d.visualization.draw_geometries([mesh])


if __name__ == "__main__":
    # test mesh properties
    print("#" * 80)
    print("Test mesh properties")
    print("#" * 80)
    for name, mesh in mesh_generator(edge_cases=True):
        check_properties(name, mesh)

    # fix triangle orientation
    print("#" * 80)
    print("Fix triangle orientation")
    print("#" * 80)
    for name, mesh in mesh_generator(edge_cases=False):
        mesh.compute_vertex_normals()
        triangles = np.asarray(mesh.get_triangles())
        rnd_idx = np.random.rand(*triangles.shape).argsort(axis=1)
        rnd_idx[0] = (0, 1, 2)
        triangles = np.take_along_axis(triangles, rnd_idx, axis=1)
        mesh.set_triangles(cv3d.utility.Vector3iVector(triangles))
        cv3d.visualization.draw_geometries([mesh])
        sucess = mesh.orient_triangles()
        print("%s orientated: %s" % (name, "yes" if sucess else "no"))
        cv3d.visualization.draw_geometries([mesh])
