# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/ransac.py

import time
import numpy as np
import cloudViewer as cv3d


def pointcloud_generator():
    # sphere = cv3d.geometry.ccMesh.create_sphere().sample_points_uniformly(int(1e4))
    # yield "sphere", sphere, 0.01, 3.0

    # mesh = cv3d.geometry.ccMesh.create_torus()
    # torus = mesh.sample_points_uniformly(int(1e4))
    # yield "torus", torus, 0.001, 3.0

    mesh = cv3d.geometry.ccMesh.create_arrow()
    arrow = mesh.sample_points_uniformly(int(1e4))
    yield "arrow", arrow, 0.001, 12.0

    d = 4
    mesh = cv3d.geometry.ccMesh.create_sphere().translate((-d, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_cone().translate((0, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_cylinder().translate((d, 0, 0))
    mesh += cv3d.geometry.ccMesh.create_torus().translate((-d, -d, 0))
    mesh += cv3d.geometry.ccMesh.create_plane().translate((0, -d, 0))
    mesh += cv3d.geometry.ccMesh.create_arrow().translate((d, -d, 0)).scale(0.5)
    unit_points = mesh.sample_points_uniformly(int(1e5))
    yield "shapes", unit_points, 0.001, 8.0

    yield "fragment", cv3d.io.read_point_cloud("../../test_data/fragment.ply"), 0.01, 6.0


if __name__ == "__main__":
    # only support on windows platform!!!

    np.random.seed(42)
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)
    random_color = True
    for pcl_name, pcl, min_radius, max_radius in pointcloud_generator():
        print("%s has %d points" % (pcl_name, pcl.size()))
        cv3d.visualization.draw_geometries([pcl])
        aabox = pcl.get_axis_aligned_bounding_box()
        scale = aabox.get_max_box_dim()
        print("cloud {} max dimension(scale) is : {}".format(pcl_name, scale))

        ransac_param = cv3d.geometry.RansacParams(scale=scale)
        enabled_list = list()
        enabled_list.append(cv3d.geometry.RansacParams.Plane)
        enabled_list.append(cv3d.geometry.RansacParams.Sphere)
        enabled_list.append(cv3d.geometry.RansacParams.Cylinder)
        enabled_list.append(cv3d.geometry.RansacParams.Cone)
        if pcl_name == "torus" or pcl_name == "shapes":
            enabled_list.append(cv3d.geometry.RansacParams.Torus)
        ransac_param.prim_enabled_list = enabled_list

        ransac_param.probability = 0.75
        ransac_param.bit_map_epsilon *= 1
        ransac_param.random_color = False
        ransac_param.support_points = 500
        ransac_param.min_radius = min_radius
        ransac_param.max_radius = max_radius
        ransac_param.max_normal_deviation_deg = 25

        start = time.time()
        ransac_result = pcl.execute_ransac(params=ransac_param, print_progress=True)
        print("execute ransac time cost : {}".format(time.time() - start))
        print("detect shape instances number: {}".format(len(ransac_result)))

        if len(ransac_result) > 0:
            out_mesh = cv3d.geometry.ccMesh()
            out_mesh.create_internal_cloud()
            out_points = cv3d.geometry.ccPointCloud("group")
            for res in ransac_result:
                prim = res.primitive
                print(prim)
                print(prim.get_name())
                if prim.is_kind_of(cv3d.geometry.ccObject.CYLINDER):
                    cylinder = cv3d.geometry.ToCylinder(prim)
                    print(cylinder.get_bottom_radius())
                elif prim.is_kind_of(cv3d.geometry.ccObject.PLANE):
                    plane = cv3d.geometry.ToPlane(prim)
                    print(plane.get_width())
                elif prim.is_kind_of(cv3d.geometry.ccObject.SPHERE):
                    sphere = cv3d.geometry.ToSphere(prim)
                    print(sphere.get_radius())
                elif prim.is_kind_of(cv3d.geometry.ccObject.CONE):
                    cone = cv3d.geometry.ToCone(prim)
                    print(cone.get_bottom_radius())
                elif prim.is_kind_of(cv3d.geometry.ccObject.TORUS):
                    torus = cv3d.geometry.ToTorus(prim)
                    print(torus.get_inside_radius())
                cloud = pcl.select_by_index(res.indices)
                if random_color:
                    color = np.random.uniform(0, 1, size=(3,))
                    cloud.paint_uniform_color(color.tolist())
                    prim.paint_uniform_color(color.tolist())
                out_points += cloud
                out_mesh += prim

            cv3d.visualization.draw_geometries([out_points])
            cv3d.visualization.draw_geometries([out_points, out_mesh], mesh_show_back_face=True)
        else:
            print("Cannot detect any shape object!")
