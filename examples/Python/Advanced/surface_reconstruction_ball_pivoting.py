# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/surface_reconstruction_ball_pivoting.py

import cloudViewer as cv3d
import numpy as np
import os

import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../Misc'))
import meshes


def problem_generator():
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    points = []
    normals = []
    for _ in range(4):
        for _ in range(4):
            pt = (np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0)
            points.append(pt)
            normals.append((0, 0, 1))
    points = np.array(points, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)
    pcd = cv3d.geometry.ccPointCloud()
    pcd.set_points(cv3d.utility.Vector3dVector(points))
    pcd.set_normals(cv3d.utility.Vector3dVector(normals))
    radii = [1, 2]
    yield pcd, radii

    cv3d.utility.set_verbosity_level(cv3d.utility.Info)

    gt_mesh = cv3d.geometry.ccMesh.create_sphere()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(100)
    radii = [0.5, 1, 2]
    yield pcd, radii

    gt_mesh = meshes.bunny()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(2000)
    radii = [0.005, 0.01, 0.02, 0.04]
    yield pcd, radii

    gt_mesh = meshes.armadillo()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(2000)
    radii = [5, 10]
    yield pcd, radii


if __name__ == "__main__":
    for pcd, radii in problem_generator():
        cv3d.visualization.draw_geometries([pcd])
        rec_mesh = cv3d.geometry.ccMesh.create_from_point_cloud_ball_pivoting(
            pcd, cv3d.utility.DoubleVector(radii))
        cv3d.visualization.draw_geometries([pcd, rec_mesh])
