# Open3selfopen3d.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/mesh_properties.py

import numpy as np
import cloudViewer as cv3d
import time

import os
import sys

if __name__ == "__main__":
    # intersection tests
    print("#" * 80)
    print("Intersection tests")
    print("#" * 80)
    np.random.seed(30)
    bbox = cv3d.geometry.ccMesh.create_box(20, 20, 20).translate((-10, -10, -10))
    meshes = [cv3d.geometry.ccMesh.create_box() for _ in range(20)]
    meshes.append(cv3d.geometry.ccMesh.create_sphere())
    meshes.append(cv3d.geometry.ccMesh.create_cone())
    meshes.append(cv3d.geometry.ccMesh.create_torus())
    dirs = [np.random.uniform(-0.1, 0.1, size=(3,)) for _ in meshes]
    for mesh in meshes:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color((0.5, 0.5, 0.5))
        mesh.translate(np.random.uniform(-7.5, 7.5, size=(3,)))
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    for mesh in meshes:
        vis.add_geometry(mesh)
    for iter in range(1000):
        for mesh, dir in zip(meshes, dirs):
            mesh.paint_uniform_color((0.5, 0.5, 0.5))
            mesh.translate(dir)
        for idx0, mesh0 in enumerate(meshes):
            collision = False
            collision = collision or mesh0.is_intersecting(bbox)
            for idx1, mesh1 in enumerate(meshes):
                if collision:
                    break
                if idx0 == idx1:
                    continue
                collision = collision or mesh0.is_intersecting(mesh1)
            if collision:
                mesh0.paint_uniform_color((1, 0, 0))
                dirs[idx0] *= -1
            vis.update_geometry(mesh0)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)
    vis.destroy_window()
