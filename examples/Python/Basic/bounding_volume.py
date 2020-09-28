# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/bounding_volume.py

import numpy as np
import cloudViewer as cv3d
import os

import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../Misc"))
import meshes

np.random.seed(42)


def mesh_generator():
    mesh = cv3d.geometry.ccMesh.create_arrow()
    mesh.compute_vertex_normals()
    mesh.rotate(mesh.get_rotation_matrix_from_xyz((0.3, 0.5, 0.1)))

    yield "rotated box mesh", mesh
    yield "rotated box pcd", mesh.sample_points_uniformly(500)

    mesh = meshes.armadillo()
    yield "armadillo mesh", mesh
    yield "armadillo pcd", mesh.sample_points_uniformly(500)


if __name__ == "__main__":
    for name, geom in mesh_generator():
        aabox = geom.get_axis_aligned_bounding_box()
        print("%s has an axis aligned box volume of %f" %
              (name, aabox.volume()))
        obox = geom.get_oriented_bounding_box()
        print("%s has an oriented box volume of %f" % (name, obox.volume()))
        aabox.set_color([1, 0, 0])
        obox.set_color([0, 1, 0])
        cv3d.visualization.draw_geometries([geom, aabox, obox])

    mesh = meshes.armadillo()
    bbox = cv3d.geometry.ccBBox(min_bound=(-30, 0, -10), max_bound=(10, 20, 10))
    cv3d.visualization.draw_geometries([mesh, bbox])
    cv3d.visualization.draw_geometries([mesh.crop(bbox), bbox])

    bbox = cv3d.geometry.ecvOrientedBBox(
        center=(-10, 10, 0),
        R=bbox.get_rotation_matrix_from_xyz((2, 1, 0)),
        extent=(40, 20, 20),
    )
    cv3d.visualization.draw_geometries([mesh, bbox])
    cv3d.visualization.draw_geometries([mesh.crop(bbox), bbox])

    pcd = mesh.sample_points_uniformly(500000)

    bbox = cv3d.geometry.ccBBox(min_bound=(-30, 0, -10), max_bound=(10, 20, 10))
    cv3d.visualization.draw_geometries([pcd, bbox])
    cv3d.visualization.draw_geometries([pcd.crop(bbox), bbox])

    obbox = cv3d.geometry.ecvOrientedBBox(
        center=(-10, 10, 0),
        R=bbox.get_rotation_matrix_from_xyz((2, 1, 0)),
        extent=(40, 20, 20),
    )
    cv3d.visualization.draw_geometries([pcd, obbox])
    cv3d.visualization.draw_geometries([pcd.crop(obbox), obbox])
