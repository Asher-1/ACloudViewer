#!/usr/bin/env python
# coding: utf-8


import cloudViewer as cv3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys

# only needed for tutorial, monkey patches visualization
sys.path.append('..')
import cloudViewer_tutorial as cv3dtut


# # Octree
# An **octree** is a tree data structure where each internal node has eight children. Octrees are commonly used for spatial partitioning of 3D point clouds. Non-empty leaf nodes of an octree contain one or more points that fall within the same spatial subdivision. Octrees are a useful description of 3D space and can be used to quickly find nearby points. CloudViewer has the geometry type `Octree` that can be used to create, search, and traverse octrees with a user-specified maximum tree depth, `max_depth`.

# ## From point cloud
# An octree can be constructed from a point cloud using the method `convert_from_point_cloud`. Each point is inserted into the tree by following the path from the root node to the appropriate leaf node at depth `max_depth`. As the tree depth increases, internal (and eventually leaf) nodes represents a smaller partition of 3D space.
# 
# If the point cloud has color, the the corresponding leaf node takes the color of the last inserted point. The `size_expand` parameter increases the size of the root octree node so it is slightly bigger than the original point cloud bounds to accomodate all points.


print('input')
N = 2000
pcd = cv3dtut.get_armadillo_mesh().sample_points_poisson_disk(N)
# fit to unit cube
pcd.scale(1 / np.max(pcd.get_max_bound() - pcd.get_min_bound()),
          center=pcd.get_center())
pcd.set_colors(cv3d.utility.Vector3dVector(np.random.uniform(0, 1, size=(N, 3))))
cv3d.visualization.draw_geometries([pcd])

print('octree division')
octree = cv3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(pcd, size_expand=0.01)
cv3d.visualization.draw_geometries([octree])


# ## From voxel grid
# An octree can also be constructed from an CloudViewer `VoxelGrid` geometry using the method `create_from_voxel_grid`. Each voxel of the input `VoxelGrid` is treated as a point in 3D space with coordinates corresponding to the origin of the voxel. Each leaf node takes the color of its corresponding voxel.

# In[ ]:


print('voxelization')
voxel_grid = cv3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
cv3d.visualization.draw_geometries([voxel_grid])

print('octree division')
octree = cv3d.geometry.Octree(max_depth=4)
octree.create_from_voxel_grid(voxel_grid)
cv3d.visualization.draw_geometries([octree])


# Additionally, an `Octree` can be coverted to a `VoxelGrid` with `to_voxel_grid`.

# ## Traversal
# An octree can be traversed which can be useful for searching or processing subsections of 3D geometry. By providing the `traverse` method with a callback, each time a node (internal or leaf) is visited, additional processing can be performed.
# 
# In the following example, an early stopping criterion is used to only process internal/leaf nodes with more than a certain number of points. This early stopping ability can be used to efficiently process spatial regions meeting certain conditions.


def f_traverse(node, node_info):
    early_stop = False

    if isinstance(node, cv3d.geometry.OctreeInternalNode):
        if isinstance(node, cv3d.geometry.OctreeInternalPointNode):
            n = 0
            for child in node.children:
                if child is not None:
                    n += 1
            print(
                "{}{}: Internal node at depth {} has {} children and {} points ({})"
                .format('    ' * node_info.depth,
                        node_info.child_index, node_info.depth, n,
                        len(node.indices), node_info.origin))

            # we only want to process nodes / spatial regions with enough points
            early_stop = len(node.indices) < 250
    elif isinstance(node, cv3d.geometry.OctreeLeafNode):
        if isinstance(node, cv3d.geometry.OctreePointColorLeafNode):
            print("{}{}: Leaf node at depth {} has {} points with origin {}".
                  format('    ' * node_info.depth, node_info.child_index,
                         node_info.depth, len(node.indices), node_info.origin))
    else:
        raise NotImplementedError('Node type not recognized!')

    # early stopping: if True, traversal of children of the current node will be skipped
    return early_stop


octree = cv3d.geometry.Octree(max_depth=4)
octree.convert_from_point_cloud(pcd, size_expand=0.01)
octree.traverse(f_traverse)


# ## Find leaf node containing point
# Using the above traversal mechanism, an octree can be quickly searched for the leaf node that contains a given point. This functionality is provided via the `locate_leaf_node` method.

octree.locate_leaf_node(pcd.get_point(0))

