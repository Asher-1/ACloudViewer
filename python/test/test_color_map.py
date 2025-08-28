import cloudViewer as cv3d
import numpy as np
import re
import os
import sys
from cloudViewer_test import download_fountain_dataset


def get_file_list(path, extension=None):

    def sorted_alphanum(file_list_ordered):
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [
            convert(c) for c in re.split('([0-9]+)', key)
        ]
        return sorted(file_list_ordered, key=alphanum_key)

    if extension is None:
        file_list = [
            path + f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ]
    else:
        file_list = [
            path + f
            for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and
            os.path.splitext(f)[1] == extension
        ]
    file_list = sorted_alphanum(file_list)
    return file_list


def test_color_map():
    path = download_fountain_dataset()
    depth_image_path = get_file_list(os.path.join(path, "depth/"),
                                     extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"),
                                     extension=".jpg")
    assert (len(depth_image_path) == len(color_image_path))

    rgbd_images = []
    for i in range(len(depth_image_path)):
        depth = cv3d.io.read_image(os.path.join(depth_image_path[i]))
        color = cv3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = cv3d.io.read_pinhole_camera_trajectory(
        os.path.join(path, "scene/key.log"))
    mesh = cv3d.io.read_triangle_mesh(
        os.path.join(path, "scene", "integrated.ply"))
    verts = np.asarray(mesh.get_vertices())
    mesh_colors = np.tile([0.40322907, 0.37276872, 0.54375919],
                          (verts.shape[0], 1))
    # mesh_colors = [0.40322907, 0.37276872, 0.54375919]
    mesh.set_vertex_colors(cv3d.utility.Vector3dVector(mesh_colors))
    # mesh.set_vertex_colors(mesh_colors)
    vertex_mean = np.mean(np.asarray(mesh.get_vertex_colors()), axis=0)
    extrinsic_mean = np.array(
        [c.extrinsic for c in camera_trajectory.parameters]).mean(axis=0)
    np.testing.assert_allclose(vertex_mean,
                               np.array([0.40322907, 0.37276872, 0.54375919]),
                               rtol=1e-2)
    np.testing.assert_allclose(
        extrinsic_mean,
        np.array([[0.77003829, -0.10813595, 0.06467495, -0.56212008],
                  [0.19100387, 0.86225833, -0.14664845, -0.81434887],
                  [-0.05557141, 0.16504166, 0.82036438, 0.27867426],
                  [0., 0., 0., 1.]]),
        rtol=1e-5)

    # Rigid Optimization
    mesh, camera_trajectory = cv3d.pipelines.color_map.run_rigid_optimizer(
        mesh, rgbd_images, camera_trajectory,
        cv3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=10))

    vertex_mean = np.mean(np.asarray(mesh.get_vertex_colors()), axis=0)
    extrinsic_mean = np.array(
        [c.extrinsic for c in camera_trajectory.parameters]).mean(axis=0)
    np.testing.assert_allclose(vertex_mean,
                               np.array([0.40294861, 0.37250299, 0.54338467]),
                               rtol=1e-5)
    np.testing.assert_allclose(
        extrinsic_mean,
        np.array([[0.7699379, -0.10768808, 0.06543989, -0.56320637],
                  [0.19119488, 0.8619734, -0.14717332, -0.8137762],
                  [-0.05608781, 0.16546427, 0.81995183, 0.27725451],
                  [0., 0., 0., 1.]]),
        rtol=1e-5)

    # Non-rigid Optimization
    mesh, camera_trajectory = cv3d.pipelines.color_map.run_non_rigid_optimizer(
        mesh, rgbd_images, camera_trajectory,
        cv3d.pipelines.color_map.NonRigidOptimizerOption(maximum_iteration=10))

    vertex_mean = np.mean(np.asarray(mesh.get_vertex_colors()), axis=0)
    extrinsic_mean = np.array(
        [c.extrinsic for c in camera_trajectory.parameters]).mean(axis=0)
    np.testing.assert_allclose(vertex_mean,
                               np.array([0.4028204, 0.37237733, 0.54322786]),
                               rtol=1e-5)
    np.testing.assert_allclose(
        extrinsic_mean,
        np.array([[0.76967962, -0.10824218, 0.0674025, -0.56381652],
                  [0.19129921, 0.86245618, -0.14634957, -0.81500831],
                  [-0.05765316, 0.16483281, 0.82054672, 0.27526268],
                  [0., 0., 0., 1.]]),
        rtol=1e-5)
