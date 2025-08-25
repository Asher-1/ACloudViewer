# cloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Advanced/color_map_optimization.py

import cloudViewer as cv3d
import os, sys

sys.path.append("../utility")
sys.path.append("../geometry")
from trajectory_io import *

from file import *

# path = "[path_to_fountain_dataset]"
path = "G:/develop/pcl_projects/cloud/dataset/tutorial/"

debug_mode = False

if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.VerbosityLevel.Debug)
    is_ci = False
    # Read RGBD images
    rgbd_images = []
    depth_image_path = get_file_list(os.path.join(path, "depth/"),
                                     extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"),
                                     extension=".png")
    assert (len(depth_image_path) == len(color_image_path))
    for i in range(len(depth_image_path)):
        depth = cv3d.io.read_image(os.path.join(depth_image_path[i]))
        color = cv3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        if debug_mode:
            pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
                rgbd_image,
                cv3d.camera.PinholeCameraIntrinsic(
                    cv3d.camera.PinholeCameraIntrinsicParameters.
                    PrimeSenseDefault))
            cv3d.visualization.draw_geometries([pcd])
        rgbd_images.append(rgbd_image)

    # Read camera pose and mesh
    camera = cv3d.io.read_pinhole_camera_trajectory(
        os.path.join(path, "scene/trajectory.log"))
    mesh = cv3d.io.read_triangle_mesh(
        os.path.join(path, "scene", "integrated.ply"))
