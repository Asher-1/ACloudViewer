# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import os

if __name__ == "__main__":

    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    print("")
    print(
        "Parameters for cv3d.pipelines.registration.PoseGraph optimization ...")
    method = cv3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = cv3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(
    )
    option = cv3d.pipelines.registration.GlobalOptimizationOption()
    print("")
    print(method)
    print(criteria)
    print(option)
    print("")

    print(
        "Optimizing Fragment cv3d.pipelines.registration.PoseGraph using cloudViewer ..."
    )

    pose_graph_data = cv3d.data.DemoPoseGraphOptimization()
    pose_graph_fragment = cv3d.io.read_pose_graph(
        pose_graph_data.pose_graph_fragment_path)
    print(pose_graph_fragment)
    cv3d.pipelines.registration.global_optimization(pose_graph_fragment, method,
                                                   criteria, option)
    cv3d.io.write_pose_graph(
        os.path.join('pose_graph_example_fragment_optimized.json'),
        pose_graph_fragment)
    print("")

    print(
        "Optimizing Global cv3d.pipelines.registration.PoseGraph using cloudViewer ..."
    )
    pose_graph_global = cv3d.io.read_pose_graph(
        pose_graph_data.pose_graph_global_path)
    print(pose_graph_global)
    cv3d.pipelines.registration.global_optimization(pose_graph_global, method,
                                                   criteria, option)
    cv3d.io.write_pose_graph(
        os.path.join('pose_graph_example_global_optimized.json'),
        pose_graph_global)
