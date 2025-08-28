# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    print("Load two aligned point clouds.")
    demo_data = cv3d.data.DemoFeatureMatchingPointClouds()
    pcd0 = cv3d.io.read_point_cloud(demo_data.point_cloud_paths[0])
    pcd1 = cv3d.io.read_point_cloud(demo_data.point_cloud_paths[1])

    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])
    cv3d.visualization.draw_geometries([pcd0, pcd1])
    print("Load their FPFH feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = cv3d.io.read_feature(demo_data.fpfh_feature_paths[0])
    feature1 = cv3d.io.read_feature(demo_data.fpfh_feature_paths[1])

    fpfh_tree = cv3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points())):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.point(i) - pcd1.point(idx[0]))
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.set_color(i, [c, c, c])
    cv3d.visualization.draw_geometries([pcd0])
    print("")

    print("Load their L32D feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = cv3d.io.read_feature(demo_data.l32d_feature_paths[0])
    feature1 = cv3d.io.read_feature(demo_data.l32d_feature_paths[1])

    fpfh_tree = cv3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.points())):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.point(i) - pcd1.point(idx[0]))
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.set_color(i, [c, c, c])
    cv3d.visualization.draw_geometries([pcd0])
    print("")
