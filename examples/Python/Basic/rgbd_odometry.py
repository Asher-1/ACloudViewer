# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/rgbd_odometry.py

import cloudViewer as cv3d
import numpy as np

if __name__ == "__main__":
    pinhole_camera_intrinsic = cv3d.io.read_pinhole_camera_intrinsic(
        "../../TestData/camera_primesense.json")
    print(pinhole_camera_intrinsic.intrinsic_matrix)

    source_color = cv3d.io.read_image("../../TestData/RGBD/color/00000.jpg")
    source_depth = cv3d.io.read_image("../../TestData/RGBD/depth/00000.png")
    target_color = cv3d.io.read_image("../../TestData/RGBD/color/00001.jpg")
    target_depth = cv3d.io.read_image("../../TestData/RGBD/depth/00001.png")
    source_rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth)
    source_pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        source_rgbd_image, pinhole_camera_intrinsic)
    target_rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth)
    target_pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        target_rgbd_image, pinhole_camera_intrinsic)

    source_pcd.paint_uniform_color([0, 1, 0])
    target_pcd.paint_uniform_color([1, 0, 0])
    print("original source pcd and target pcd...")
    cv3d.visualization.draw_geometries([source_pcd, target_pcd])

    option = cv3d.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)

    [success_color_term, trans_color_term, info] = cv3d.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
         odo_init, cv3d.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term, info] = cv3d.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image, pinhole_camera_intrinsic,
         odo_init, cv3d.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        source_pcd_color_term = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_color_term.transform(trans_color_term)
        cv3d.visualization.draw_geometries([target_pcd, source_pcd_color_term])
    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        cv3d.visualization.draw_geometries([target_pcd, source_pcd_hybrid_term])
