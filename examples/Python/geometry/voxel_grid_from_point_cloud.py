# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np

if __name__ == "__main__":

    N = 2000
    armadillo_data = cv3d.data.ArmadilloMesh()
    pcd = cv3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(N)
    # Fit to unit cube.
    pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
              center=pcd.get_center())
    pcd.set_colors(cv3d.utility.Vector3dVector(np.random.uniform(0, 1,
                                                              size=(N, 3))))
    print('Displaying input point cloud ...')
    cv3d.visualization.draw([pcd])

    print('Displaying voxel grid ...')
    voxel_grid = cv3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                                voxel_size=0.05)
    cv3d.visualization.draw([voxel_grid])
