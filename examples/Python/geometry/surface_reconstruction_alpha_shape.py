# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/surface_reconstruction_alpha_shape.py

import cloudViewer as cv3d
import numpy as np
import os
import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../Misc"))
import meshes


def draw_geometries_with_back_face(geometries):
    visualizer = cv3d.visualization.Visualizer()
    visualizer.create_window()
    render_option = visualizer.get_render_option()
    render_option.mesh_show_back_face = True
    for geometry in geometries:
        visualizer.add_geometry(geometry)
    visualizer.run()
    visualizer.destroy_window()


if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    mesh = meshes.bunny()
    pcd = mesh.sample_points_poisson_disk(750)
    cv3d.visualization.draw_geometries([pcd])
    for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
        print("alpha={}".format(alpha))
        mesh = cv3d.geometry.ccMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        mesh.compute_vertex_normals()
        draw_geometries_with_back_face([mesh])

    pcd = cv3d.io.read_point_cloud("../../TestData/fragment.ply")
    cv3d.visualization.draw_geometries([pcd])
    print("compute tetra mesh only once")
    tetra_mesh, pt_map = cv3d.geometry.TetraMesh.create_from_point_cloud(pcd)
    print("done with tetra mesh")
    for alpha in np.logspace(np.log10(0.5), np.log10(0.01), num=4):
        print("alpha={}".format(alpha))
        mesh = cv3d.geometry.ccMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha, tetra_mesh, pt_map)
        mesh.compute_vertex_normals()
        draw_geometries_with_back_face([mesh])
