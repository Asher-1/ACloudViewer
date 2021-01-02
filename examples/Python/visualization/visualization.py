# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/visualization.py

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = cv3d.io.read_point_cloud("../../TestData/fragment.ply")
    cv3d.visualization.draw_geometries([pcd])

    print("Let's draw some primitives")
    mesh_box = cv3d.geometry.ccMesh.create_box(width=1.0,
                                               height=1.0,
                                               depth=1.0)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    mesh_sphere = cv3d.geometry.ccMesh.create_sphere(radius=1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = cv3d.geometry.ccMesh.create_cylinder(radius=0.3,
                                                         height=4.0)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = cv3d.geometry.ccMesh.create_coordinate_frame(
        size=0.6, origin=[-2, -2, -2])

    print("We draw a few primitives using collection.")
    cv3d.visualization.draw_geometries(
        [mesh_box, mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    cv3d.visualization.draw_geometries(
        [mesh_box + mesh_sphere + mesh_cylinder + mesh_frame])

    print("Let's draw a cubic using cv3d.geometry.LineSet.")
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 7],
        [6, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = cv3d.geometry.LineSet(
        points=cv3d.utility.Vector3dVector(points),
        lines=cv3d.utility.Vector2iVector(lines),
    )
    line_set.colors = cv3d.utility.Vector3dVector(colors)
    cv3d.visualization.draw_geometries([line_set])

    print("Let's draw a textured triangle mesh from obj file.")
    textured_mesh = cv3d.io.read_triangle_mesh("../../TestData/crate/crate.obj")
    textured_mesh.compute_vertex_normals()
    cv3d.visualization.draw_geometries([textured_mesh])
