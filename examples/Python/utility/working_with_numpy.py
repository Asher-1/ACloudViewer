# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/working_with_numpy.py

import copy
import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":

    # generate some neat n times 3 matrix using a variant of sync function
    x = np.linspace(-3, 3, 401)
    mesh_x, mesh_y = np.meshgrid(x, x)
    z = np.sinc((np.power(mesh_x, 2) + np.power(mesh_y, 2)))
    z_norm = (z - z.min()) / (z.max() - z.min())
    xyz = np.zeros((np.size(mesh_x), 3))
    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_norm, -1)
    print('xyz')
    print(xyz)

    # Pass xyz to cloudViewer.cv3d.geometry.ccPointCloud and visualize
    pcd = cv3d.geometry.ccPointCloud()
    pcd.set_points(cv3d.utility.Vector3dVector(xyz))
    cv3d.io.write_point_cloud("../../TestData/sync.ply", pcd)

    # Load saved point cloud and visualize it
    pcd_load = cv3d.io.read_point_cloud("../../TestData/sync.ply")
    cv3d.visualization.draw_geometries([pcd_load])

    # convert cloudViewer.cv3d.geometry.ccPointCloud to numpy array
    xyz_load = np.asarray(pcd_load.get_points())
    print('xyz_load')
    print(xyz_load)

    # save z_norm as an image (change [0,1] range to [0,255] range with uint8 type)
    img = cv3d.geometry.Image((z_norm * 255).astype(np.uint8))
    cv3d.io.write_image("../../TestData/sync.png", img)
    cv3d.visualization.draw_geometries([img])
