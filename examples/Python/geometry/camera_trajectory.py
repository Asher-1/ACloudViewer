# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/camera_trajectory.py

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
    trajectory = cv3d.io.read_pinhole_camera_trajectory(
        "../../TestData/RGBD/trajectory.log")
    cv3d.io.write_pinhole_camera_trajectory("test.json", trajectory)
    print(trajectory)
    print(trajectory.parameters[0].extrinsic)
    print(np.asarray(trajectory.parameters[0].extrinsic))
    for i in range(5):
        im1 = cv3d.io.read_image(
            "../../TestData/RGBD/depth/{:05d}.png".format(i))
        im2 = cv3d.io.read_image(
            "../../TestData/RGBD/color/{:05d}.jpg".format(i))
        im = cv3d.geometry.RGBDImage.create_from_color_and_depth(
            im2, im1, 1000.0, 5.0, False)
        pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
            im, trajectory.parameters[i].intrinsic,
            trajectory.parameters[i].extrinsic)
        pcds.append(pcd)
    cv3d.visualization.draw_geometries(pcds)
    print("")
