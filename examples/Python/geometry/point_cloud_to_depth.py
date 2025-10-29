# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    tum_data = cv3d.data.SampleTUMRGBDImage()
    depth = cv3d.t.io.read_image(tum_data.depth_path)
    intrinsic = cv3d.core.Tensor([[535.4, 0, 320.1], [0, 539.2, 247.6],
                                  [0, 0, 1]])

    pcd = cv3d.t.geometry.PointCloud.create_from_depth_image(depth,
                                                             intrinsic,
                                                             depth_scale=5000.0,
                                                             depth_max=10.0)
    cv3d.visualization.draw([pcd])
    depth_reproj = pcd.project_to_depth_image(640,
                                              480,
                                              intrinsic,
                                              depth_scale=5000.0,
                                              depth_max=10.0)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(np.asarray(depth.to_legacy()))
    axs[1].imshow(np.asarray(depth_reproj.to_legacy()))
    plt.show()
