# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/Python/Benchmark/benchmark_pre.py

import os
import sys
import pickle
import cloudViewer as cv3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from cloudViewer_example import *

do_visualization = True


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        cv3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = cv3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        cv3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature,
                                              max_nn=100))
    return pcd_down, pcd_fpfh


if __name__ == "__main__":
    # data preparation
    dataset = cv3d.data.LivingRoomPointClouds()
    n_ply_files = len(dataset.paths)
    voxel_size = 0.05

    alignment = []
    for s in range(n_ply_files):
        source = cv3d.io.read_point_cloud(dataset.paths[s])
        source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
