# cloudViewer: Asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Basic/rgbd_nyu.py

import cloudViewer as cv3d
import numpy as np
import re
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    img = np.frombuffer(buffer,
                        dtype=byteorder + 'u2',
                        count=int(width) * int(height),
                        offset=len(header)).reshape((int(height), int(width)))
    img_out = img.astype('u2')
    return img_out


if __name__ == "__main__":
    print("Read NYU dataset")
    # cloudViewer does not support ppm/pgm file yet. Not using cv3d.io.read_image here.
    # MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
    color_raw = mpimg.imread("../../test_data/RGBD/other_formats/NYU_color.ppm")
    depth_raw = read_nyu_pgm("../../test_data/RGBD/other_formats/NYU_depth.pgm")
    color = cv3d.geometry.Image(color_raw)
    depth = cv3d.geometry.Image(depth_raw)
    rgbd_image = cv3d.geometry.RGBDImage.create_from_nyu_format(color, depth)
    print(rgbd_image)
    plt.subplot(1, 2, 1)
    plt.title('NYU grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('NYU depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()
    pcd = cv3d.geometry.ccPointCloud.create_from_rgbd_image(
        rgbd_image,
        cv3d.camera.PinholeCameraIntrinsic(
            cv3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # Flip it, otherwise the ccPointCloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    cv3d.visualization.draw_geometries([pcd])
