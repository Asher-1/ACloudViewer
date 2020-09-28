# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/rgbd_tum.py

import cloudViewer as cv3d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Read TUM dataset")
    color_raw = cv3d.io.read_image(
        "../../TestData/RGBD/other_formats/TUM_color.png")
    depth_raw = cv3d.io.read_image(
        "../../TestData/RGBD/other_formats/TUM_depth.png")
    rgbd_image = cv3d.geometry.RGBDImage.create_from_tum_format(color_raw, depth_raw)
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('TUM grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('TUM depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        rgbd_image,
        cv3d.camera.PinholeCameraIntrinsic(
            cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the ccPointCloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cv3d.visualization.draw_geometries([pcd])
