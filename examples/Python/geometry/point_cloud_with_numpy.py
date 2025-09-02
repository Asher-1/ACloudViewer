# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":
    # Generate some n x 3 matrix using a variant of sync function.
    x = np.linspace(-3, 3, 201)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print("Printing numpy array used to make CloudViewer pointcloud ...")
    print(xyz)

    # Pass xyz to cv3d.geometry.ccPointCloud and visualize.
    pcd = cv3d.geometry.ccPointCloud()
    pcd.set_points(cv3d.utility.Vector3dVector(xyz))
    # Add color and estimate normals for better visualization.
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(1)
    print("Displaying CloudViewer pointcloud made using numpy array ...")
    cv3d.visualization.draw([pcd])

    # Convert cv3d.geometry.ccPointCloud to numpy array.
    xyz_converted = np.asarray(pcd.get_points())
    print("Printing numpy array made using CloudViewer pointcloud ...")
    print(xyz_converted)
