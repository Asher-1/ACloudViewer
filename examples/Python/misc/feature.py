# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Misc/feature.py

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    print("Load two aligned point clouds.")
    pcd0 = cv3d.io.read_point_cloud("../../test_data/Feature/cloud_bin_0.pcd")
    pcd1 = cv3d.io.read_point_cloud("../../test_data/Feature/cloud_bin_1.pcd")
    pcd0.paint_uniform_color([1, 0.706, 0])
    pcd1.paint_uniform_color([0, 0.651, 0.929])
    cv3d.visualization.draw_geometries([pcd0, pcd1])
    print("Load their FPFH feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = cv3d.io.read_feature(
        "../../test_data/Feature/cloud_bin_0.fpfh.bin")
    feature1 = cv3d.io.read_feature(
        "../../test_data/Feature/cloud_bin_1.fpfh.bin")
    fpfh_tree = cv3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.get_points())):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.get_point(i) - pcd1.get_point(idx[0]))
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.set_color(i, [c, c, c])
    cv3d.visualization.draw_geometries([pcd0])
    print("")

    print("Load their L32D feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = cv3d.io.read_feature("../../test_data/Feature/cloud_bin_0.d32.bin")
    feature1 = cv3d.io.read_feature("../../test_data/Feature/cloud_bin_1.d32.bin")
    fpfh_tree = cv3d.geometry.KDTreeFlann(feature1)
    for i in range(len(pcd0.get_points())):
        [_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.get_point(i) - pcd1.get_point(idx[0]))
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.set_color(i, [c, c, c])
    cv3d.visualization.draw_geometries([pcd0])
    print("")
