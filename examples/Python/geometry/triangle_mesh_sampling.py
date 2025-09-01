# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import time
import cloudViewer as cv3d


def time_fcn(fcn, *fcn_args, runs=5):
    times = []
    for _ in range(runs):
        tic = time.time()
        res = fcn(*fcn_args)
        times.append(time.time() - tic)
    return res, times


def mesh_generator():
    yield cv3d.geometry.ccMesh.create_plane()
    yield cv3d.geometry.ccMesh.create_sphere()

    bunny = cv3d.data.BunnyMesh()
    mesh = cv3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()
    yield mesh


if __name__ == "__main__":
    plane = cv3d.geometry.ccMesh.create_plane()
    cv3d.visualization.draw_geometries([plane])

    print('Uniform sampling can yield clusters of points on the surface')
    pcd = plane.sample_points_uniformly(number_of_points=500)
    cv3d.visualization.draw_geometries([pcd])

    print(
        'Poisson disk sampling can evenly distributes the points on the surface.'
    )
    print('The method implements sample elimination.')
    print('Therefore, the method starts with a sampled point cloud and removes '
          'point to satisfy the sampling criterion.')
    print('The method supports two options to provide the initial point cloud')
    print('1) Default via the parameter init_factor: The method first samples '
          'uniformly a point cloud from the mesh with '
          'init_factor x number_of_points and uses this for the elimination')
    pcd = plane.sample_points_poisson_disk(number_of_points=500, init_factor=5)
    cv3d.visualization.draw_geometries([pcd])

    print(
        '2) one can provide an own point cloud and pass it to the '
        'cv3d.geometry.sample_points_poisson_disk method. Then this point cloud is used '
        'for elimination.')
    print('Initial point cloud')
    pcd = plane.sample_points_uniformly(number_of_points=2500)
    cv3d.visualization.draw_geometries([pcd])
    pcd = plane.sample_points_poisson_disk(number_of_points=500, pcl=pcd)
    cv3d.visualization.draw_geometries([pcd])

    print('Timings')
    for mesh in mesh_generator():
        mesh.compute_vertex_normals()
        cv3d.visualization.draw_geometries([mesh])

        pcd, times = time_fcn(mesh.sample_points_uniformly, 500)
        print('sample uniform took on average: %f[s]' % np.mean(times))
        cv3d.visualization.draw_geometries([pcd])

        pcd, times = time_fcn(mesh.sample_points_poisson_disk, 500, 5)
        print('sample poisson disk took on average: %f[s]' % np.mean(times))
        cv3d.visualization.draw_geometries([pcd])
