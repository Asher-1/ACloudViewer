# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import PIL.Image
import IPython.display

interactive = True


def jupyter_draw_geometries(
    geoms,
    window_name="CloudViewer",
    width=1920,
    height=1080,
    left=50,
    top=50,
    point_show_normal=False,
    mesh_show_wireframe=False,
    mesh_show_back_face=False,
    lookat=None,
    up=None,
    front=None,
    zoom=None,
):
    vis = cv3d.visualization.Visualizer()
    vis.create_window(
        window_name=window_name,
        width=width,
        height=height,
        left=left,
        top=top,
        visible=True,  # If false, capture_screen_float_buffer() won't work.
    )
    vis.get_render_option().point_show_normal = point_show_normal
    vis.get_render_option().mesh_show_wireframe = mesh_show_wireframe
    vis.get_render_option().mesh_show_back_face = mesh_show_back_face
    for geom in geoms:
        vis.add_geometry(geom)
    if lookat is not None:
        vis.get_view_control().set_lookat(lookat)
    if up is not None:
        vis.get_view_control().set_up(up)
    if front is not None:
        vis.get_view_control().set_front(front)
    if zoom is not None:
        vis.get_view_control().set_zoom(zoom)
    if interactive:
        vis.run()
    else:
        for geom in geoms:
            vis.update_geometry(geom)
        vis.poll_events()
        vis.update_renderer()
    im = vis.capture_screen_float_buffer()
    vis.destroy_window()
    im = (255 * np.asarray(im)).astype(np.uint8)
    IPython.display.display(PIL.Image.fromarray(im, "RGB"))


cv3d.visualization.draw_geometries = jupyter_draw_geometries

# cv3d.visualization.draw = jupyter_draw_geometries


def edges_to_lineset(mesh, edges, color):
    ls = cv3d.geometry.LineSet()
    ls.points = mesh.vertices()
    ls.lines = edges
    ls.paint_uniform_color(color)
    return ls


def get_non_manifold_edge_mesh():
    verts = np.array(
        [[-1, 0, 0], [0, 1, 0], [1, 0, 0], [0, -1, 0], [0, 0, 1]],
        dtype=np.float64,
    )
    triangles = np.array([[0, 1, 3], [1, 2, 3], [1, 3, 4]])
    mesh = cv3d.geometry.ccMesh()
    mesh.create_internal_cloud()
    mesh.set_vertices(cv3d.utility.Vector3dVector(verts))
    mesh.set_triangles(cv3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    mesh.rotate(
        mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)),
        center=mesh.get_center(),
    )
    return mesh


def get_non_manifold_vertex_mesh():
    verts = np.array(
        [
            [-1, 0, -1],
            [1, 0, -1],
            [0, 1, -1],
            [0, 0, 0],
            [-1, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.float64,
    )
    triangles = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [1, 2, 3],
        [2, 0, 3],
        [4, 5, 6],
        [4, 5, 3],
        [5, 6, 3],
        [4, 6, 3],
    ])
    mesh = cv3d.geometry.ccMesh()
    mesh.create_internal_cloud()
    mesh.set_vertices(cv3d.utility.Vector3dVector(verts))
    mesh.set_triangles(cv3d.utility.Vector3iVector(triangles))
    mesh.compute_vertex_normals()
    mesh.rotate(
        mesh.get_rotation_matrix_from_xyz((np.pi / 4, 0, np.pi / 4)),
        center=mesh.get_center(),
    )
    return mesh


def get_open_box_mesh():
    mesh = cv3d.geometry.ccMesh.create_box()
    mesh.set_triangles(
        cv3d.utility.Vector3iVector(np.asarray(mesh.get_triangles())[:-2]))
    mesh.compute_vertex_normals()
    mesh.rotate(
        mesh.get_rotation_matrix_from_xyz((0.8 * np.pi, 0, 0.66 * np.pi)),
        center=mesh.get_center(),
    )
    return mesh


def get_intersecting_boxes_mesh():
    mesh0 = cv3d.geometry.ccMesh.create_box()
    T = np.eye(4)
    T[:, 3] += (0.5, 0.5, 0.5, 0)
    mesh1 = cv3d.geometry.ccMesh.create_box()
    mesh1.transform(T)
    mesh = mesh0 + mesh1
    mesh.compute_vertex_normals()
    mesh.rotate(
        mesh.get_rotation_matrix_from_xyz((0.7 * np.pi, 0, 0.6 * np.pi)),
        center=mesh.get_center(),
    )
    return mesh
