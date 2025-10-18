# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np

if __name__ == "__main__":
    pinhole_camera_intrinsic = cv3d.camera.PinholeCameraIntrinsic(
        cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    rgbd_data = cv3d.data.SampleRedwoodRGBDImages()
    source_color = cv3d.io.read_image(rgbd_data.color_paths[0])
    source_depth = cv3d.io.read_image(rgbd_data.depth_paths[0])
    target_color = cv3d.io.read_image(rgbd_data.color_paths[1])
    target_depth = cv3d.io.read_image(rgbd_data.depth_paths[1])

    source_rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth)
    target_rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth)
    target_pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        target_rgbd_image, pinhole_camera_intrinsic)

    option = cv3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)

    [success_color_term, trans_color_term,
     info] = cv3d.pipelines.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image,
         pinhole_camera_intrinsic, odo_init,
         cv3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term,
     info] = cv3d.pipelines.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image,
         pinhole_camera_intrinsic, odo_init,
         cv3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        source_pcd_color_term = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_color_term.transform(trans_color_term)
        cv3d.visualization.draw([target_pcd, source_pcd_color_term])

    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        cv3d.visualization.draw([target_pcd, source_pcd_hybrid_term])
