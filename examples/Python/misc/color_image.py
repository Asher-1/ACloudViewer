# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Misc/color_image.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# conda install pillow matplotlib

import cloudViewer as cv3d

if __name__ == "__main__":

    print("Testing image in cloudViewer ...")
    print("Convert an image to numpy and draw it with matplotlib.")
    x = cv3d.io.read_image("../../TestData/image.PNG")
    print(x)
    plt.imshow(np.asarray(x))
    plt.show()

    print(
        "Convet a numpy image to cv3d.geometry.Image and show it with DrawGeomtries()."
    )
    y = mpimg.imread("../../TestData/lena_color.jpg")
    print(y.shape)
    yy = cv3d.geometry.Image(y)
    print(yy)
    cv3d.visualization.draw_geometries([yy])

    print("Render a channel of the previous image.")
    z = np.array(y[:, :, 1])
    print(z.shape)
    print(z.strides)
    zz = cv3d.geometry.Image(z)
    print(zz)
    cv3d.visualization.draw_geometries([zz])

    print("Write the previous image to file then load it with matplotlib.")
    cv3d.io.write_image("test.jpg", zz, quality=100)
    zzz = mpimg.imread("test.jpg")
    plt.imshow(zzz)
    plt.show()

    print("Testing basic image processing module.")
    im_raw = mpimg.imread("../../TestData/lena_color.jpg")
    im = cv3d.geometry.Image(im_raw)
    im_g3 = im.filter(cv3d.geometry.ImageFilterType.Gaussian3)
    im_g5 = im.filter(cv3d.geometry.ImageFilterType.Gaussian5)
    im_g7 = im.filter(cv3d.geometry.ImageFilterType.Gaussian7)
    im_gaussian = [im, im_g3, im_g5, im_g7]
    pyramid_levels = 4
    pyramid_with_gaussian_filter = True
    im_pyramid = im.create_pyramid(pyramid_levels, pyramid_with_gaussian_filter)
    im_dx = im.filter(cv3d.geometry.ImageFilterType.Sobel3dx)
    im_dx_pyramid = cv3d.geometry.Image.filter_pyramid(
        im_pyramid, cv3d.geometry.ImageFilterType.Sobel3dx)
    im_dy = im.filter(cv3d.geometry.ImageFilterType.Sobel3dy)
    im_dy_pyramid = cv3d.geometry.Image.filter_pyramid(
        im_pyramid, cv3d.geometry.ImageFilterType.Sobel3dy)
    switcher = {
        0: im_gaussian,
        1: im_pyramid,
        2: im_dx_pyramid,
        3: im_dy_pyramid,
    }
    for i in range(4):
        for j in range(pyramid_levels):
            plt.subplot(4, pyramid_levels, i * 4 + j + 1)
            plt.imshow(switcher.get(i)[j])
    plt.show()

    print("")
