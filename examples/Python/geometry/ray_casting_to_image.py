# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import cloudViewer as cv3d
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Create meshes and convert to cloudViewer.t.geometry.TriangleMesh .
    cube = cv3d.geometry.TriangleMesh.create_box().translate([0, 0, 0])
    cube = cv3d.t.geometry.TriangleMesh.from_legacy(cube)
    torus = cv3d.geometry.TriangleMesh.create_torus().translate([0, 0, 2])
    torus = cv3d.t.geometry.TriangleMesh.from_legacy(torus)
    sphere = cv3d.geometry.TriangleMesh.create_sphere(radius=0.5).translate(
        [1, 2, 3])
    sphere = cv3d.t.geometry.TriangleMesh.from_legacy(sphere)

    scene = cv3d.t.geometry.RaycastingScene()
    scene.add_triangles(cube)
    scene.add_triangles(torus)
    _ = scene.add_triangles(sphere)

    rays = cv3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=[0, 0, 2],
        eye=[2, 3, 0],
        up=[0, 1, 0],
        width_px=640,
        height_px=480,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = scene.cast_rays(rays)
    plt.imshow(ans['t_hit'].numpy())
    plt.show()
    plt.imshow(np.abs(ans['primitive_normals'].numpy()))
    plt.show()
    plt.imshow(np.abs(ans['geometry_ids'].numpy()), vmax=3)
    plt.show()
