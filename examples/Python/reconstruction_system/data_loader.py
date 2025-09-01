# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d


def lounge_data_loader():
    print('Loading Stanford Lounge RGB-D Dataset')

    # Get the dataset.
    lounge_rgbd = cv3d.data.LoungeRGBDImages()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = lounge_rgbd.extract_dir
    config['path_intrinsic'] = ""
    config['depth_max'] = 3.0
    config['voxel_size'] = 0.05
    config['depth_diff_max'] = 0.07
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 3.0
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config


def bedroom_data_loader():
    print('Loading Redwood Bedroom RGB-D Dataset')

    # Get the dataset.
    bedroom_rgbd = cv3d.data.BedroomRGBDImages()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = bedroom_rgbd.extract_dir
    config['path_intrinsic'] = ""
    config['depth_max'] = 3.0
    config['voxel_size'] = 0.05
    config['depth_diff_max'] = 0.07
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 3.0
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config


def jackjack_data_loader():
    print('Loading RealSense L515 Jack-Jack RGB-D Bag Dataset')

    # Get the dataset.
    jackjack_bag = cv3d.data.JackJackL515Bag()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = jackjack_bag.path
    config['path_intrinsic'] = ""
    config['depth_max'] = 0.85
    config['voxel_size'] = 0.025
    config['depth_diff_max'] = 0.03
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 0.75
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config
