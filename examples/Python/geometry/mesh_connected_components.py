# cloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Basic/mesh_connected_components.py

import cloudViewer as cv3d
import numpy as np
import copy
import time
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../misc"))
import meshes

if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    print("Generate data")
    mesh = meshes.bunny()

    print("Subdivide mesh to make it a bit harder")
    mesh = mesh.subdivide_midpoint(number_of_iterations=2)
    print(mesh)

    vert = np.asarray(mesh.get_vertices())
    min_vert, max_vert = vert.min(axis=0), vert.max(axis=0)
    for _ in range(30):
        cube = cv3d.geometry.ccMesh.create_box()
        cube.scale(0.005)
        cube.translate(
            (
                np.random.uniform(min_vert[0], max_vert[0]),
                np.random.uniform(min_vert[1], max_vert[1]),
                np.random.uniform(min_vert[2], max_vert[2]),
            ),
            relative=False,
        )
        mesh += cube
    mesh.compute_vertex_normals()

    print("Cluster connected triangles")
    tic = time.time()
    triangle_clusters, cluster_n_triangles, cluster_area = (
        mesh.cluster_connected_triangles())
    print("  took {}[s]".format(time.time() - tic))
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    print("Show input mesh")
    cv3d.visualization.draw_geometries([mesh])

    print("Show mesh with small clusters removed")
    mesh_0 = copy.deepcopy(mesh)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    cv3d.visualization.draw_geometries([mesh_0])

    print("Show largest cluster")
    mesh_1 = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_1.remove_triangles_by_mask(triangles_to_remove)
    cv3d.visualization.draw_geometries([mesh_1])
