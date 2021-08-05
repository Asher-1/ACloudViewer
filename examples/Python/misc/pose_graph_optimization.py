# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Misc/pose_graph_optimization.py

import cloudViewer as cv3d
import numpy as np

if __name__ == "__main__":

    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    print("")
    print("Parameters for cv3d.registration.PoseGraph optimization ...")
    method = cv3d.registration.GlobalOptimizationLevenbergMarquardt()
    criteria = cv3d.registration.GlobalOptimizationConvergenceCriteria()
    option = cv3d.registration.GlobalOptimizationOption()
    print("")
    print(method)
    print(criteria)
    print(option)
    print("")

    print("Optimizing Fragment cv3d.registration.PoseGraph using cloudViewer ...")
    data_path = "../../test_data/GraphOptimization/"
    pose_graph_fragment = cv3d.io.read_pose_graph(data_path + "pose_graph_example_fragment.json")
    print(pose_graph_fragment)
    cv3d.registration.global_optimization(pose_graph_fragment, method, criteria, option)
    cv3d.io.write_pose_graph(
        data_path + "pose_graph_example_fragment_optimized.json",
        pose_graph_fragment)
    print("")

    print("Optimizing Global cv3d.registration.PoseGraph using cloudViewer ...")
    pose_graph_global = cv3d.io.read_pose_graph(data_path + "pose_graph_example_global.json")
    print(pose_graph_global)
    cv3d.registration.global_optimization(pose_graph_global, method, criteria, option)
    cv3d.io.write_pose_graph(
        data_path + "pose_graph_example_global_optimized.json",
        pose_graph_global)
    print("")
