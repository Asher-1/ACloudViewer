# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Misc/evaluate_geometric_feature.py

import cloudViewer as cv3d
import numpy as np


def evaluate(pcd_target, pcd_source, feature_target, feature_source):
    tree_target = cv3d.geometry.KDTreeFlann(feature_target)
    pt_dis = np.zeros(len(pcd_source.get_points()))
    for i in range(len(pcd_source.get_points())):
        [_, idx, _] = tree_target.search_knn_vector_xd(feature_source.data[:, i], 1)
        pt_dis[i] = np.linalg.norm(pcd_source.get_point(i) - pcd_target.get_point(idx[0]))
    return pt_dis


if __name__ == "__main__":
    pcd_target = cv3d.io.read_point_cloud(
        "../../TestData/Feature/cloud_bin_0.pcd")
    pcd_source = cv3d.io.read_point_cloud(
        "../../TestData/Feature/cloud_bin_1.pcd")
    feature_target = cv3d.io.read_feature(
        "../../TestData/Feature/cloud_bin_0.fpfh.bin")
    feature_source = cv3d.io.read_feature(
        "../../TestData/Feature/cloud_bin_1.fpfh.bin")
    pt_dis = evaluate(pcd_target, pcd_source, feature_target, feature_source)
    num_good = sum(pt_dis < 0.075)
    print(
        "{:.2f}% points in source ccPointCloud successfully found their correspondence."
        .format(num_good * 100.0 / len(pcd_source.get_points())))
