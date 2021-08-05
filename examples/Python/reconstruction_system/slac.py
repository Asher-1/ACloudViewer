# Open3D: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/python/reconstruction_system/slac.py

import numpy as np
import cloudViewer as cv3d
import sys
sys.path.append("../utility")
from file import join, get_file_list, write_poses_to_log

sys.path.append(".")


def run(config):
    print("slac non-rigid optimisation.")
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)

    path_dataset = config['path_dataset']

    ply_file_names = get_file_list(
        join(config["path_dataset"], config["folder_fragment"]), ".ply")

    if (len(ply_file_names) == 0):
        raise RuntimeError(
            "No fragment found in {}, please make sure the reconstruction_system has finished running on the dataset."
            .format(join(config["path_dataset"], config["folder_fragment"])))

    pose_graph_fragment = cv3d.io.read_pose_graph(
        join(path_dataset, config["template_refined_posegraph_optimized"]))

    # SLAC optimizer parameters.
    slac_params = cv3d.t.pipelines.slac.slac_optimizer_params(
        max_iterations=config["max_iterations"],
        voxel_size=config["voxel_size"],
        distance_threshold=config["distance_threshold"],
        fitness_threshold=config["fitness_threshold"],
        regularizer_weight=config["regularizer_weight"],
        device=cv3d.core.Device(str(config["device"])),
        slac_folder=path_dataset + config["folder_slac"])

    # SLAC debug option.
    debug_option = cv3d.t.pipelines.slac.slac_debug_option(False, 0)

    # Run the system.
    pose_graph_updated = cv3d.pipelines.registration.PoseGraph()

    # rigid optimization method.
    if (config["method"] == "rigid"):
        pose_graph_updated = cv3d.t.pipelines.slac.run_rigid_optimizer_for_fragments(
            ply_file_names, pose_graph_fragment, slac_params, debug_option)
    elif (config["method"] == "slac"):
        pose_graph_updated, ctrl_grid = cv3d.t.pipelines.slac.run_slac_optimizer_for_fragments(
            ply_file_names, pose_graph_fragment, slac_params, debug_option)

        hashmap = ctrl_grid.get_hashmap()
        active_addrs = hashmap.get_active_addrs().to(cv3d.core.Dtype.Int64)

        key_tensor = hashmap.get_key_tensor()[active_addrs]
        key_tensor.save(
            join(slac_params.get_subfolder_name(), "ctr_grid_keys.npy"))

        value_tensor = hashmap.get_value_tensor()[active_addrs]
        value_tensor.save(
            join(slac_params.get_subfolder_name(), "ctr_grid_values.npy"))

    else:
        raise RuntimeError(
            "Requested optimization method {}, is not implemented. Implemented methods includes slac and rigid."
            .format(config["method"]))

    # Write updated pose graph.
    cv3d.io.write_pose_graph(
        join(slac_params.get_subfolder_name(),
             config["template_optimized_posegraph_slac"]), pose_graph_updated)

    # Write trajectory for slac-integrate stage.
    fragment_folder = join(path_dataset, config["folder_fragment"])
    params = []
    for i in range(len(pose_graph_updated.nodes)):
        fragment_pose_graph = cv3d.io.read_pose_graph(
            join(fragment_folder, "fragment_optimized_%03d.json" % i))
        for node in fragment_pose_graph.nodes:
            pose = np.dot(pose_graph_updated.nodes[i].pose, node.pose)
            param = cv3d.camera.PinholeCameraParameters()
            param.extrinsic = np.linalg.inv(pose)
            params.append(param)

    trajectory = cv3d.camera.PinholeCameraTrajectory()
    trajectory.parameters = params

    cv3d.io.write_pinhole_camera_trajectory(
        slac_params.get_subfolder_name() + "/optimized_trajectory_" +
        str(config["method"]) + ".log", trajectory)
