# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/mesh_simplification.py

import numpy as np
import cloudViewer as cv3d

import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../misc'))
import meshes


def mesh_generator():
    mesh = meshes.plane()
    yield mesh.subdivide_midpoint(2)

    mesh = cv3d.geometry.ccMesh.create_box()
    yield mesh.subdivide_midpoint(2)

    mesh = cv3d.geometry.ccMesh.create_sphere()
    yield mesh.subdivide_midpoint(2)

    mesh = cv3d.geometry.ccMesh.create_cone()
    yield mesh.subdivide_midpoint(2)

    mesh = cv3d.geometry.ccMesh.create_cylinder()
    yield mesh.subdivide_midpoint(2)

    yield meshes.bathtub()

    yield meshes.bunny()


if __name__ == "__main__":
    np.random.seed(42)

    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        n_verts = np.asarray(mesh.get_vertices()).shape[0]
        mesh.set_vertex_colors(cv3d.utility.Vector3dVector(
            np.random.uniform(0, 1, size=(n_verts, 3))))

        print("original mesh has %d triangles and %d vertices" % (np.asarray(
            mesh.get_triangles()).shape[0], np.asarray(mesh.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh])

        voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 4
        target_number_of_triangles = np.asarray(mesh.get_triangles()).shape[0] // 2
        print('voxel_size = %f' % voxel_size)

        mesh_smp = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=cv3d.geometry.SimplificationContraction.Average)
        print(
            "vertex clustered mesh (average) has %d triangles and %d vertices" %
            (np.asarray(mesh_smp.get_triangles()).shape[0],
             np.asarray(mesh_smp.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh_smp])

        mesh_smp = mesh.simplify_vertex_clustering(
            voxel_size=voxel_size,
            contraction=cv3d.geometry.SimplificationContraction.Quadric)
        print(
            "vertex clustered mesh (quadric) has %d triangles and %d vertices" %
            (np.asarray(mesh_smp.get_triangles()).shape[0],
             np.asarray(mesh_smp.get_vertices()).shape[0]))
        cv3d.visualization.draw_geometries([mesh_smp])

        mesh_smp = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_number_of_triangles)
        smp_triangles = np.asarray(mesh_smp.get_triangles())
        smp_vertices = np.asarray(mesh_smp.get_vertices())
        print("quadric decimated mesh has %d triangles and %d vertices" %
              (smp_triangles.shape[0], smp_vertices.shape[0]))
        cv3d.visualization.draw_geometries([mesh_smp])
