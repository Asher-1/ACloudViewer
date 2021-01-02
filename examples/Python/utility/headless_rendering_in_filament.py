#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  tutorial
FILE_NAME    :  headless_rendering_in_filament
AUTHOR       :  DAHAI LU
TIME         :  2020/12/31 上午10:36
PRODUCT_NAME :  PyCharm
================================================================
"""

import cloudViewer as cv3d
import cloudViewer.visualization.rendering as rendering


if __name__ == '__main__':

    box = cv3d.geometry.ccMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    render = cv3d.visualization.rendering.OffscreenRenderer(640, 480)
    grey = rendering.Material()
    grey.base_color = [0.7, 0.7, 0.7, 1.0]
    grey.shader = "defaultLit"
    render.scene.add_geometry("box", box, grey)
    render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    img = render.render_to_image()
    cv3d.visualization.draw_geometries([img])