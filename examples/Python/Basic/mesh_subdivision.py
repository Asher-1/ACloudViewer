# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/mesh_subdivision.py

import numpy as np
import cloudViewer as cv3d

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../Misc'))
import meshes


def mesh_generator():
    yield meshes.triangle()
    yield meshes.plane()
    yield cv3d.geometry.ccMesh.create_tetrahedron()
    yield cv3d.geometry.ccMesh.create_box()
    yield cv3d.geometry.ccMesh.create_octahedron()
    yield cv3d.geometry.ccMesh.create_icosahedron()
    yield cv3d.geometry.ccMesh.create_sphere()
    yield cv3d.geometry.ccMesh.create_cone()
    yield cv3d.geometry.ccMesh.create_cylinder()
    yield meshes.knot()
    yield meshes.bathtub()


if __name__ == "__main__":
    np.random.seed(42)

    number_of_iterations = 3

    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        n_verts = np.asarray(mesh.get_vertices()).shape[0]
        colors = np.random.uniform(0, 1, size=(n_verts, 3))
        mesh.set_vertex_colors(cv3d.utility.Vector3dVector(colors))

        print("original mesh has %d triangles and %d vertices" % (np.asarray(
            mesh.get_triangles()).shape[0], np.asarray(mesh.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh])

        mesh_up = mesh.subdivide_midpoint(
            number_of_iterations=number_of_iterations)
        print("midpoint upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.get_triangles()).shape[0],
               np.asarray(mesh_up.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh_up])

        mesh_up = mesh.subdivide_loop(number_of_iterations=number_of_iterations)
        print("loop upsampled mesh has %d triangles and %d vertices" %
              (np.asarray(mesh_up.get_triangles()).shape[0],
               np.asarray(mesh_up.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh_up])
