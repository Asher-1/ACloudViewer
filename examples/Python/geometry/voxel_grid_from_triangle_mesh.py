# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as cv3d
import numpy as np

if __name__ == "__main__":
    bunny = cv3d.data.BunnyMesh()
    mesh = cv3d.io.read_triangle_mesh(bunny.path)
    mesh.compute_vertex_normals()

    # Fit to unit cube.
    mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()),
               center=mesh.get_center())
    print('Displaying input mesh ...')
    cv3d.visualization.draw([mesh])

    voxel_grid = cv3d.geometry.VoxelGrid.create_from_triangle_mesh(
        mesh, voxel_size=0.05)
    print('Displaying voxel grid ...')
    cv3d.visualization.draw([voxel_grid])
