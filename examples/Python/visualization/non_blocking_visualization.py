# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np


def prepare_data():
    pcd_data = cv3d.data.DemoICPPointClouds()
    source_raw = cv3d.io.read_point_cloud(pcd_data.paths[0])
    target_raw = cv3d.io.read_point_cloud(pcd_data.paths[1])
    source = source_raw.voxel_down_sample(voxel_size=0.02)
    target = target_raw.voxel_down_sample(voxel_size=0.02)

    trans = [[0.862, 0.011, -0.507, 0.0], [-0.139, 0.967, -0.215, 0.7],
             [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]]
    source.transform(trans)
    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    target.transform(flip_transform)
    return source, target


def demo_non_blocking_visualization():
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    source, target = prepare_data()
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    vis.add_geometry(target)
    threshold = 0.05
    icp_iteration = 100
    save_image = False

    for i in range(icp_iteration):
        reg_p2l = cv3d.pipelines.registration.registration_icp(
            source, target, threshold, np.identity(4),
            cv3d.pipelines.registration.TransformationEstimationPointToPlane(),
            cv3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1))
        source.transform(reg_p2l.transformation)
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()
        if save_image:
            vis.capture_screen_image("temp_%04d.jpg" % i)
    vis.destroy_window()

    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Info)


if __name__ == '__main__':
    demo_non_blocking_visualization()
