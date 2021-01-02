# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/surface_reconstruction_poisson.py

import cloudViewer as cv3d
import numpy as np
import matplotlib.pyplot as plt
import os

import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../misc"))
import meshes

if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    pcd = meshes.eagle()
    print(pcd)
    cv3d.visualization.draw_geometries([pcd])

    print('run Poisson surface reconstruction')
    mesh, densities = cv3d.geometry.ccMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh.compute_vertex_normals()
    cv3d.io.write_triangle_mesh("poisson_eagle.ply", mesh)
    print(mesh)
    cv3d.visualization.draw_geometries([mesh])

    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = cv3d.geometry.ccMesh()
    density_mesh.create_internal_cloud()
    density_mesh.set_vertices(mesh.get_vertices())
    density_mesh.set_triangles(mesh.get_triangles())
    density_mesh.set_triangle_normals(mesh.get_triangle_normals())
    density_mesh.set_vertex_colors(cv3d.utility.Vector3dVector(density_colors))
    cv3d.visualization.draw_geometries([density_mesh])

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    cv3d.visualization.draw_geometries([mesh])
