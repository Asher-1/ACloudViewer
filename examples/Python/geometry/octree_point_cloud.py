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
    pcd.set_colors(
        cv3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3))))
    print('Displaying input pointcloud ...')
    cv3d.visualization.draw([pcd])

    octree = cv3d.geometry.Octree(max_depth=4)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    print(octree)
    print('Displaying octree ..')
    cv3d.visualization.draw([octree])
