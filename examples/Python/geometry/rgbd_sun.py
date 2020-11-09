# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/rgbd_sun.py

import cloudViewer as cv3d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Read SUN dataset")
    color_raw = cv3d.io.read_image(
        "../../TestData/RGBD/other_formats/SUN_color.jpg")
    depth_raw = cv3d.io.read_image(
        "../../TestData/RGBD/other_formats/SUN_depth.png")
    rgbd_image = cv3d.geometry.RGBDImage.create_from_sun_format(
        color_raw, depth_raw)
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('SUN grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('SUN depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        rgbd_image,
        cv3d.camera.PinholeCameraIntrinsic(
            cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the ccPointCloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cv3d.visualization.draw_geometries([pcd])
