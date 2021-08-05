# CloudViewer: www.erow.cn
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/ReconstructionSystem/debug/pairwise_rgbd_alignment.py

import argparse
import json
import sys
import cloudViewer as cv3d
sys.path.append("../utility")
from file import *
from visualization import *
sys.path.append(".")
from initialize_config import *
from make_fragments import *


def test_single_pair(s, t, color_files, depth_files, intrinsic, with_opencv,
                     config):
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)
    [success, trans,
     info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                                    with_opencv, config)
    print(trans)
    print(info)
    print(intrinsic)
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], False,
                                        config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], False,
                                        config)
    source = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        source_rgbd_image, intrinsic)
    target = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        target_rgbd_image, intrinsic)
    draw_geometries_flip([source, target])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mathching two RGBD images")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("source_id", type=int, help="ID of source RGBD image")
    parser.add_argument("target_id", type=int, help="ID of target RGBD image")
    parser.add_argument("--path_intrinsic",
                        help="path to the RGBD camera intrinsic")
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)
        initialize_config(config)

        with_opencv = initialize_opencv()
        if with_opencv:
            from opencv_pose_estimation import pose_estimation

        [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
        if args.path_intrinsic:
            intrinsic = cv3d.io.read_pinhole_camera_intrinsic(
                args.path_intrinsic)
        else:
            intrinsic = cv3d.camera.PinholeCameraIntrinsic(
                cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        test_single_pair(args.source_id, args.target_id, color_files,
                         depth_files, intrinsic, with_opencv, config)
