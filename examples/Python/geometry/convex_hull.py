# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/convex_hull.py

import cloudViewer as cv3d

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../Misc'))
import meshes


def mesh_generator():
    yield cv3d.geometry.ccMesh.create_box()
    yield cv3d.geometry.ccMesh.create_sphere()
    yield meshes.knot()
    yield meshes.bunny()
    yield meshes.armadillo()


if __name__ == "__main__":
    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        hull, _ = mesh.compute_convex_hull()
        hull_ls = cv3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        cv3d.visualization.draw_geometries([mesh, hull_ls])

        pcl = mesh.sample_points_poisson_disk(number_of_points=2000)
        hull, _ = pcl.compute_convex_hull()
        hull_ls = cv3d.geometry.LineSet.create_from_triangle_mesh(hull)
        hull_ls.paint_uniform_color((1, 0, 0))
        cv3d.visualization.draw_geometries([pcl, hull_ls])
