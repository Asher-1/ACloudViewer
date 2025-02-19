# ----------------------------------------------------------------------------
# -                        CloudViewer: Asher-1.github.io                    -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 Asher-1.github.io
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import time
import pytest
import os
from cloudViewer_test import test_data_dir

_eight_cubes_colors = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.1, 0.1, 0.0],
    [0.0, 0.0, 0.1],
    [0.1, 0.0, 0.1],
    [0.0, 0.1, 0.1],
    [0.1, 0.1, 0.1],
])

_eight_cubes_points = np.array([
    [0.5, 0.5, 0.5],
    [1.5, 0.5, 0.5],
    [0.5, 1.5, 0.5],
    [1.5, 1.5, 0.5],
    [0.5, 0.5, 1.5],
    [1.5, 0.5, 1.5],
    [0.5, 1.5, 1.5],
    [1.5, 1.5, 1.5],
])


def test_octree_OctreeNodeInfo():
    origin = [0, 0, 0]
    size = 2.0
    depth = 5
    child_index = 7

    node_info = cv3d.geometry.OctreeNodeInfo(origin, size, depth, child_index)
    np.testing.assert_equal(node_info.origin, origin)
    np.testing.assert_equal(node_info.size, size)
    np.testing.assert_equal(node_info.depth, depth)
    np.testing.assert_equal(node_info.child_index, child_index)


def test_octree_OctreeColorLeafNode():
    color_leaf_node = cv3d.geometry.OctreeColorLeafNode()
    color = [0.1, 0.2, 0.3]
    color_leaf_node.color = color
    np.testing.assert_equal(color_leaf_node.color, color)

    # Test copy constructor
    color_leaf_node_copy = cv3d.geometry.OctreeColorLeafNode(color_leaf_node)
    np.testing.assert_equal(color_leaf_node_copy.color, color)

    # Test OctreeLeafNode's inherited operator== function
    assert color_leaf_node == color_leaf_node_copy
    assert color_leaf_node_copy == color_leaf_node

    # Test OctreeLeafNode's inherited clone() function
    color_leaf_node_clone = color_leaf_node.clone()
    np.testing.assert_equal(color_leaf_node_clone.color, color)
    assert color_leaf_node == color_leaf_node_clone
    assert color_leaf_node_clone == color_leaf_node


def test_octree_init():
    octree = cv3d.geometry.Octree(1, [0, 0, 0], 2)


def test_octree_convert_from_point_cloud():
    octree = cv3d.geometry.Octree(1, [0, 0, 0], 2)

    pcd = cv3d.geometry.ccPointCloud()
    pcd.set_points(cv3d.utility.Vector3dVector(_eight_cubes_points))
    pcd.set_colors(cv3d.utility.Vector3dVector(_eight_cubes_colors))
    octree.convert_from_point_cloud(pcd)


def test_octree_insert_point():
    octree = cv3d.geometry.Octree(1, [0, 0, 0], 2)
    for point, color in zip(_eight_cubes_points, _eight_cubes_colors):
        f_init = cv3d.geometry.OctreeColorLeafNode.get_init_function()
        f_update = cv3d.geometry.OctreeColorLeafNode.get_update_function(color)
        octree.insert_point(point, f_init, f_update)


def test_octree_node_access():
    octree = cv3d.geometry.Octree(1, [0, 0, 0], 2)
    for point, color in zip(_eight_cubes_points, _eight_cubes_colors):
        f_init = cv3d.geometry.OctreeColorLeafNode.get_init_function()
        f_update = cv3d.geometry.OctreeColorLeafNode.get_update_function(color)
        octree.insert_point(point, f_init, f_update)
    for i in range(8):
        np.testing.assert_equal(octree.root_node.children[i].color,
                                _eight_cubes_colors[i])


def test_octree_visualize():
    pcd_path = os.path.join(test_data_dir, "fragment.ply")
    pcd = cv3d.io.read_point_cloud(pcd_path)
    octree = cv3d.geometry.Octree(8)
    octree.convert_from_point_cloud(pcd)
    # Enable the following line to test visualization
    # cv3d.visualization.draw_geometries([octree])


def test_octree_voxel_grid_convert():
    pcd_path = os.path.join(test_data_dir, "fragment.ply")
    pcd = cv3d.io.read_point_cloud(pcd_path)
    octree = cv3d.geometry.Octree(8)
    octree.convert_from_point_cloud(pcd)

    voxel_grid = octree.to_voxel_grid()
    octree_copy = voxel_grid.to_octree(max_depth=8)

    # Enable the following line to test visualization
    # cv3d.visualization.draw_geometries([octree])
    # cv3d.visualization.draw_geometries([voxel_grid])
    # cv3d.visualization.draw_geometries([octree_copy])


def test_locate_leaf_node():
    pcd_path = os.path.join(test_data_dir, "fragment.ply")
    pcd = cv3d.io.read_point_cloud(pcd_path)

    max_depth = 5
    octree = cv3d.geometry.Octree(max_depth)
    octree.convert_from_point_cloud(pcd, 0.01)

    # Try locating a few points
    for idx in range(0, len(pcd.get_points()), 200):
        point = pcd.get_points()[idx]
        node, node_info = octree.locate_leaf_node(np.array(point))
        # The located node must be in bound
        assert octree.is_point_in_bound(point, node_info.origin, node_info.size)
        # Leaf node must be located
        assert node_info.depth == max_depth
        # Leaf node's size must match
        assert node_info.size == octree.size / np.power(2, max_depth)
