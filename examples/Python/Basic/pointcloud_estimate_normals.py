# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/pointcloud_estimate_normals.py

import numpy as np
import cloudViewer as cv3d
import time
import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '../Misc'))
import meshes

np.random.seed(42)


def knn_generator():
    yield 'knn20', cv3d.geometry.KDTreeSearchParamKNN(20)
    yield 'radius', cv3d.geometry.KDTreeSearchParamRadius(0.01666)
    yield 'hybrid', cv3d.geometry.KDTreeSearchParamHybrid(0.01666, 60)


def pointcloud_generator():
    pts = np.random.uniform(-30, 30, size=(int(1e5), 3))
    pcl = cv3d.geometry.ccPointCloud()
    pcl.set_points(cv3d.utility.Vector3dVector(pts))
    yield 'uniform', pcl

    yield 'moebius', cv3d.geometry.ccMesh.create_moebius().sample_points_uniformly(int(1e5))

    yield 'bunny', meshes.bunny().scale(10).sample_points_uniformly(int(1e5))

    yield 'fragment', cv3d.io.read_point_cloud('../../TestData/fragment.ply')


if __name__ == "__main__":
    # Benchmark
    for pcl_name, pcl in pointcloud_generator():
        for knn_name, knn in knn_generator():
            print('-' * 80)
            for fast_normal_computation in [True, False]:
                times = []
                for _ in range(50):
                    tic = time.time()
                    pcl.estimate_normals(
                        knn, fast_normal_computation=fast_normal_computation)
                    times.append(time.time() - tic)
                print('fast={}: {}, {} -- avg time={}[s]'.format(
                    fast_normal_computation, pcl_name, knn_name,
                    np.mean(times)))

    # Test
    for pcl_name, pcl in pointcloud_generator():
        for knn_name, knn in knn_generator():
            pcl.estimate_normals(knn, True)
            normals_fast = np.asarray(pcl.get_normals())
            pcl.estimate_normals(knn, False)
            normals_eigen = np.asarray(pcl.get_normals())
            test = (normals_eigen * normals_fast).sum(axis=1)
            print('normals agree: {}'.format(np.all(test - 1 < 1e-9)))

    # Test normals of flat surface
    X, Y = np.mgrid[0:1:0.1, 0:1:0.1]
    X = X.flatten()
    Y = Y.flatten()

    pts = np.zeros((3, X.size))
    pts[0] = X
    pts[1] = Y

    shape = cv3d.geometry.ccPointCloud()
    shape.set_points(cv3d.utility.Vector3dVector(pts.T))
    shape.paint_uniform_color([0, 0.651, 0.929])  # blue

    shape.estimate_normals(cv3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30),
                           fast_normal_computation=True)
    cv3d.visualization.draw_geometries([shape])

    shape.estimate_normals(cv3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30),
                           fast_normal_computation=False)
    cv3d.visualization.draw_geometries([shape])
