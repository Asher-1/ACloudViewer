# cloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Advanced/rgbd_integration.py

import cloudViewer as cv3d

import sys
sys.path.append("../utility")
sys.path.append("../geometry")
from trajectory_io import *
import numpy as np

if __name__ == "__main__":
    camera_poses = read_trajectory("../../test_data/RGBD/odometry.log")
    volume = cv3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=4.0 / 512.0,
        sdf_trunc=0.04,
        color_type=cv3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    for i in range(len(camera_poses)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = cv3d.io.read_image(
            "../../test_data/RGBD/color/{:05d}.jpg".format(i))
        depth = cv3d.io.read_image(
            "../../test_data/RGBD/depth/{:05d}.png".format(i))
        rgbd = cv3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            cv3d.camera.PinholeCameraIntrinsic(
                cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
            np.linalg.inv(camera_poses[i].pose))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cv3d.visualization.draw_geometries([mesh])
