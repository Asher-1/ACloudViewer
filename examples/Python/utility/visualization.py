# CloudViewer: asher-1.github.io
# The MIT License (MIT)
# See license file or visit Asher-1.github.io for details

# examples/Python/Utility/visualization.py

import copy
import cloudViewer as cv3d

flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]


def draw_geometries_flip(pcds):
    pcds_transform = []
    for pcd in pcds:
        pcd_temp = copy.deepcopy(pcd)
        pcd_temp.transform(flip_transform)
        pcds_transform.append(pcd_temp)
    cv3d.visualization.draw_geometries(pcds_transform)


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    source_temp.transform(flip_transform)
    target_temp.transform(flip_transform)
    cv3d.visualization.draw_geometries([source_temp, target_temp])


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.transform(transformation)
    source_temp.transform(flip_transform)
    target_temp.transform(flip_transform)
    cv3d.visualization.draw_geometries([source_temp, target_temp])
