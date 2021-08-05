# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/kdtree.py

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":
    print("Testing kdtree in cloudViewer ...")
    print("Load a point cloud and paint it gray.")
    pcd = cv3d.io.read_point_cloud("../../test_data/Feature/cloud_bin_0.pcd")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = cv3d.geometry.KDTreeFlann(pcd)

    print("Paint the 1500th point red.")
    pcd.set_color(1500, [1, 0, 0])

    print("Find its 200 nearest neighbors, paint blue.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.get_point(1500), 200)
    colors = np.asarray(pcd.get_colors())
    colors[idx[1:], :] = [0, 0, 1]
    pcd.set_colors(cv3d.utility.Vector3dVector(colors))

    print("Find its neighbors with distance less than 0.2, paint green.")
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.get_point(1500), 0.2)
    colors = np.asarray(pcd.get_colors())
    colors[idx[1:], :] = [0, 1, 0]
    pcd.set_colors(cv3d.utility.Vector3dVector(colors))

    print("Visualize the point cloud.")
    cv3d.visualization.draw_geometries([pcd],
                                       zoom=0.5599,
                                       front=[-0.4958, 0.8229, 0.2773],
                                       lookat=[2.1126, 1.0163, -1.8543],
                                       up=[0.1007, -0.2626, 0.9596])
    print("")
