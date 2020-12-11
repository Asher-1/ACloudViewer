# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Utility/transformation.py

import numpy as np
import cloudViewer as cv3d
import time


def geometry_generator():
    mesh = cv3d.geometry.ccMesh.create_sphere()
    verts = np.asarray(mesh.get_vertices())
    colors = np.random.uniform(0, 1, size=verts.shape)
    mesh.set_vertex_colors(cv3d.utility.Vector3dVector(colors))
    mesh.compute_vertex_normals()

    pcl = cv3d.geometry.ccPointCloud()
    pcl.set_points(mesh.get_vertices())
    pcl.set_colors(mesh.get_vertex_colors())
    pcl.set_normals(mesh.get_vertex_normals())
    yield pcl

    yield cv3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    yield mesh


def animate(geom):
    vis = cv3d.visualization.Visualizer()
    vis.create_window()

    geom.rotate(geom.get_rotation_matrix_from_xyz((0.75, 0.5, 0)))
    vis.add_geometry(geom)

    scales = [0.9 for _ in range(30)] + [1 / 0.9 for _ in range(30)]
    axisangles = [(0.2 / np.sqrt(2), 0.2 / np.sqrt(2), 0) for _ in range(60)]
    ts = [(0.1, 0.1, -0.1) for _ in range(30)
         ] + [(-0.1, -0.1, 0.1) for _ in range(30)]

    for scale, aa in zip(scales, axisangles):
        R = geom.get_rotation_matrix_from_axis_angle(aa)
        geometry = geom.scale(scale)
        geometry.rotate(R, geometry.get_center())
        vis.update_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    for t in ts:
        geom.translate(t)
        vis.update_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    for scale, aa, t in zip(scales, axisangles, ts):
        R = geom.get_rotation_matrix_from_axis_angle(aa)
        geom.scale(scale).translate(t).rotate(R)
        vis.update_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)


if __name__ == "__main__":
    for geom in geometry_generator():
        animate(geom)
