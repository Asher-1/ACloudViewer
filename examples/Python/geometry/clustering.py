# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/clustering.py

import numpy as np
import cloudViewer as cv3d
import matplotlib.pyplot as plt

np.random.seed(42)


def pointcloud_generator():
    yield "sphere", cv3d.geometry.ccMesh.create_sphere().\
        sample_points_uniformly(int(1e4)), 0.4

    mesh = cv3d.geometry.ccMesh.create_torus()
    # mesh.scale(5, center=mesh.get_geometry_center())
    mesh.scale(5)
    mesh += cv3d.geometry.ccMesh.create_torus()
    yield "torus", mesh.sample_points_uniformly(int(1e4)), 0.75

    d = 4
    mesh = cv3d.geometry.ccMesh.create_tetrahedron().translate((-d, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_octahedron().translate((0, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_icosahedron().translate((d, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_torus().translate((-d, -d, 0))
    mesh += cv3d.geometry.ccMesh.create_moebius(twists=1).translate((0, -d, 0))
    mesh += cv3d.geometry.ccMesh.create_moebius(twists=2).translate((d, -d, 0))
    yield "shapes", mesh.sample_points_uniformly(int(1e5)), 0.5

    yield "fragment", cv3d.io.read_point_cloud("../../TestData/fragment.ply"), 0.02


if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)

    cmap = plt.get_cmap("tab20")
    for pcl_name, pcl, eps in pointcloud_generator():
        print("%s has %d points" % (pcl_name, pcl.size()))
        cv3d.visualization.draw_geometries([pcl])

        labels = np.array(
            pcl.cluster_dbscan(eps=eps, min_points=10, print_progress=True))
        max_label = labels.max()
        print("%s has %d clusters" % (pcl_name, max_label + 1))

        colors = cmap(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcl.set_colors(cv3d.utility.Vector3dVector(colors[:, :3]))
        cv3d.visualization.draw_geometries([pcl])
