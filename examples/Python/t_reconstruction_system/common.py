# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/t_reconstruction_system/common.py

import cloudViewer as cv3d

import os
import sys
import json
import numpy as np
import glob
from os.path import isfile, join, splitext, dirname, basename
from warnings import warn


def extract_rgbd_frames(rgbd_video_file):
    """
    Extract color and aligned depth frames and intrinsic calibration from an
    RGBD video file (currently only RealSense bag files supported). Folder
    structure is:
        <directory of rgbd_video_file/<rgbd_video_file name without extension>/
            {depth/00000.jpg,color/00000.png,intrinsic.json}
    """
    frames_folder = join(dirname(rgbd_video_file),
                         basename(splitext(rgbd_video_file)[0]))
    path_intrinsic = join(frames_folder, "intrinsic.json")
    if isfile(path_intrinsic):
        warn(f"Skipping frame extraction for {rgbd_video_file} since files are"
             " present.")
    else:
        rgbd_video = cv3d.t.io.RGBDVideoReader.create(rgbd_video_file)
        rgbd_video.save_frames(frames_folder)
    with open(path_intrinsic) as intr_file:
        intr = json.load(intr_file)
    depth_scale = intr["depth_scale"]
    return frames_folder, path_intrinsic, depth_scale


def lounge_dataloader(config):
    # Get the dataset.
    lounge_rgbd = cv3d.data.LoungeRGBDImages()
    # Override default config parameters with dataset specific parameters.
    config.path_dataset = lounge_rgbd.extract_dir
    config.path_trajectory = lounge_rgbd.trajectory_log_path
    config.depth_folder = "depth"
    config.color_folder = "color"
    return config


def bedroom_dataloader(config):
    # Get the dataset.
    bedroom_rgbd = cv3d.data.BedroomRGBDImages()
    # Override default config parameters with dataset specific parameters.
    config.path_dataset = bedroom_rgbd.extract_dir
    config.path_trajectory = bedroom_rgbd.trajectory_log_path
    config.depth_folder = "depth"
    config.color_folder = "image"
    return config


def jack_jack_dataloader(config):
    # Get the dataset.
    jackjack_rgbd = cv3d.data.JackJackL515Bag()
    # Override default config parameters with dataset specific parameters.
    print("Extracting frames from RGBD video file")
    config.path_dataset = jackjack_rgbd.path
    config.depth_folder = "depth"
    config.color_folder = "color"
    return config


def get_default_dataset(config):
    print('Config file was not provided, falling back to default dataset.')
    if config.default_dataset == 'lounge':
        config = lounge_dataloader(config)
    elif config.default_dataset == 'bedroom':
        config = bedroom_dataloader(config)
    elif config.default_dataset == 'jack_jack':
        config = jack_jack_dataloader(config)
    else:
        print(
            "The requested dataset is not available. Available dataset options include lounge and jack_jack."
        )
        sys.exit(1)

    print('Loaded data from {}'.format(config.path_dataset))
    return config


def load_depth_file_names(config):
    if not os.path.exists(config.path_dataset):
        print(
            f"Path '{config.path_dataset}' not found.",
            'Please provide --path_dataset in the command line or the config file.'
        )
        return [], []

    depth_folder = os.path.join(config.path_dataset, config.depth_folder)

    # Only 16-bit png depth is supported
    depth_file_names = glob.glob(os.path.join(depth_folder, '*.png'))
    n_depth = len(depth_file_names)
    if n_depth == 0:
        print(f'Depth image not found in {depth_folder}, abort!')
        return []

    return sorted(depth_file_names)


def load_rgbd_file_names(config):
    depth_file_names = load_depth_file_names(config)
    if len(depth_file_names) == 0:
        return [], []

    color_folder = os.path.join(config.path_dataset, config.color_folder)
    extensions = ['*.png', '*.jpg']
    for ext in extensions:
        color_file_names = glob.glob(os.path.join(color_folder, ext))
        if len(color_file_names) == len(depth_file_names):
            return depth_file_names, sorted(color_file_names)

    depth_folder = os.path.join(config.path_dataset, config.depth_folder)
    print('Found {} depth images in {}, but cannot find matched number of '
          'color images in {} with extensions {}, abort!'.format(
              len(depth_file_names), depth_folder, color_folder, extensions))
    return [], []


def load_intrinsic(config, key='depth'):
    path_intrinsic = config.path_color_intrinsic if key == 'color' else config.path_intrinsic

    if path_intrinsic is None or path_intrinsic == '':
        intrinsic = cv3d.camera.PinholeCameraIntrinsic(
            cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    else:
        intrinsic = cv3d.io.read_pinhole_camera_intrinsic(path_intrinsic)

    if config.engine == 'legacy':
        return intrinsic
    elif config.engine == 'tensor':
        return cv3d.core.Tensor(intrinsic.intrinsic_matrix,
                               cv3d.core.Dtype.Float64)
    else:
        print('Unsupported engine {}'.format(config.engine))


def load_extrinsics(path_trajectory, config):
    extrinsics = []

    # For either a fragment or a scene
    if path_trajectory.endswith('log'):
        data = cv3d.io.read_pinhole_camera_trajectory(path_trajectory)
        for param in data.parameters:
            extrinsics.append(param.extrinsic)

    # Only for a fragment
    elif path_trajectory.endswith('json'):
        data = cv3d.io.read_pose_graph(path_trajectory)
        for node in data.nodes:
            extrinsics.append(np.linalg.inv(node.pose))

    if config.engine == 'legacy':
        return extrinsics
    elif config.engine == 'tensor':
        return list(
            map(lambda x: cv3d.core.Tensor(x, cv3d.core.Dtype.Float64),
                extrinsics))
    else:
        print('Unsupported engine {}'.format(config.engine))


def save_poses(
    path_trajectory,
    poses,
    intrinsic=cv3d.camera.PinholeCameraIntrinsic(
        cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)):
    if path_trajectory.endswith('log'):
        traj = cv3d.camera.PinholeCameraTrajectory()
        params = []
        for pose in poses:
            param = cv3d.camera.PinholeCameraParameters()
            param.intrinsic = intrinsic
            param.extrinsic = np.linalg.inv(pose)
            params.append(param)
        traj.parameters = params
        cv3d.io.write_pinhole_camera_trajectory(path_trajectory, traj)

    elif path_trajectory.endswith('json'):
        pose_graph = cv3d.pipelines.registration.PoseGraph()
        for pose in poses:
            node = cv3d.pipelines.registration.PoseGraphNode()
            node.pose = pose
            pose_graph.nodes.append(node)
        cv3d.io.write_pose_graph(path_trajectory, pose_graph)


def extract_pointcloud(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()

        pcd = mesh.get_associated_cloud()

        if file_name is not None:
            cv3d.io.write_point_cloud(file_name, pcd)

    elif config.engine == 'tensor':
        pcd = volume.extract_point_cloud(
            weight_threshold=config.surface_weight_thr)

        if file_name is not None:
            cv3d.io.write_point_cloud(file_name, pcd.to_legacy())

    return pcd


def extract_trianglemesh(volume, config, file_name=None):
    if config.engine == 'legacy':
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        if file_name is not None:
            cv3d.io.write_triangle_mesh(file_name, mesh)

    elif config.engine == 'tensor':
        mesh = volume.extract_triangle_mesh(
            weight_threshold=config.surface_weight_thr)
        mesh = mesh.to_legacy()

        if file_name is not None:
            cv3d.io.write_triangle_mesh(file_name, mesh)

    return mesh
