# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/ccPointCloud.py

import numpy as np
import cloudViewer as cv3d

if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)
    print("Load a ply point cloud, print it, and render it")
    pcd = cv3d.io.read_point_cloud("../../TestData/fragment.ply")
    print(pcd)
    print(pcd.size())
    print(pcd.has_points())
    print(pcd.has_normals())
    print(pcd.has_colors())
    print(np.asarray(pcd.get_points()))
    cv3d.visualization.draw_geometries([pcd])

    pcd.set_temp_color([0, 0, 1])
    pcd.set_opacity(0.5)
    pcd.show_colors(True)
    print(pcd.get_opacity())
    print(pcd.get_temp_color())
    print(pcd.is_color_overriden())
    cv3d.visualization.draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = pcd.voxel_down_sample(voxel_size=0.05)
    cv3d.visualization.draw_geometries([downpcd])

    print("Recompute the normal of the downsampled point cloud")
    downpcd.estimate_normals(search_param=cv3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))
    cv3d.visualization.draw_geometries([downpcd])

    print("Print a normal vector of the 0th point")
    print(downpcd.get_normal(0))
    print("Print the normal vectors of the first 10 points")
    print(np.asarray(downpcd.get_normals())[:10, :])
    print("")

    print("Load a polygon volume and use it to crop the original point cloud")
    vol = cv3d.visualization.read_selection_polygon_volume(
        "../../TestData/Crop/cropped.json")
    chair = vol.crop_point_cloud(pcd)
    cv3d.visualization.draw_geometries([chair])
    print("")

    print("Paint chair")
    chair.paint_uniform_color([1, 0.706, 0])
    cv3d.visualization.draw_geometries([chair])
    print("")
