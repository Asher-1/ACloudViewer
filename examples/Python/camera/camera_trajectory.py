# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    print("Testing camera in cloudViewer ...")
    intrinsic = cv3d.camera.PinholeCameraIntrinsic(
        cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    print(intrinsic.intrinsic_matrix)
    print(cv3d.camera.PinholeCameraIntrinsic())
    x = cv3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    print(x)
    print(x.intrinsic_matrix)
    cv3d.io.write_pinhole_camera_intrinsic("test.json", x)
    y = cv3d.io.read_pinhole_camera_intrinsic("test.json")
    print(y)
    print(np.asarray(y.intrinsic_matrix))

    print("Read a trajectory and combine all the RGB-D images.")
    pcds = []
    redwood_rgbd = cv3d.data.SampleRedwoodRGBDImages()
    trajectory = cv3d.io.read_pinhole_camera_trajectory(
        redwood_rgbd.trajectory_log_path)
    cv3d.io.write_pinhole_camera_trajectory("test.json", trajectory)
    print(trajectory)
    print(trajectory.parameters[0].extrinsic)
    print(np.asarray(trajectory.parameters[0].extrinsic))
    for i in range(5):
        im1 = cv3d.io.read_image(redwood_rgbd.depth_paths[i])
        im2 = cv3d.io.read_image(redwood_rgbd.color_paths[i])
        im = cv3d.geometry.RGBDImage.create_from_color_and_depth(
            im2, im1, 1000.0, 5.0, False)
        pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
            im, trajectory.parameters[i].intrinsic,
            trajectory.parameters[i].extrinsic)
        pcds.append(pcd)
    cv3d.visualization.draw_geometries(pcds)
