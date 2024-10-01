#!/usr/bin/env python

import cloudViewer as cv3d
import cloudViewer.visualization.rendering as rendering


def main():
    render = rendering.OffscreenRenderer(640, 480)

    yellow = rendering.Material()
    yellow.base_color = [1.0, 0.75, 0.0, 1.0]
    yellow.shader = "defaultLit"

    green = rendering.Material()
    green.base_color = [0.0, 0.5, 0.0, 1.0]
    green.shader = "defaultLit"

    grey = rendering.Material()
    grey.base_color = [0.7, 0.7, 0.7, 1.0]
    grey.shader = "defaultLit"

    white = rendering.Material()
    white.base_color = [1.0, 1.0, 1.0, 1.0]
    white.shader = "defaultLit"

    cyl = cv3d.geometry.ccMesh.create_cylinder(.05, 3)
    cyl.compute_vertex_normals()
    cyl.translate([-2, 0, 1.5])
    sphere = cv3d.geometry.ccMesh.create_sphere(.2)
    sphere.compute_vertex_normals()
    sphere.translate([-2, 0, 3])

    box = cv3d.geometry.ccMesh.create_box(2, 2, 1)
    box.compute_vertex_normals()
    box.translate([-1, -1, 0])
    solid = cv3d.geometry.ccMesh.create_icosahedron(0.5)
    solid.compute_triangle_normals()
    solid.compute_vertex_normals()
    solid.translate([0, 0, 1.75])

    render.scene.add_geometry("cyl", cyl, green)
    render.scene.add_geometry("sphere", sphere, yellow)
    render.scene.add_geometry("box", box, grey)
    render.scene.add_geometry("solid", solid, white)
    render.scene.camera.look_at([0, 0, 0], [0, 10, 0], [0, 0, 1])
    render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0],
                                     75000)
    render.scene.scene.enable_sun_light(True)
    render.scene.show_axes(True)

    img = render.render_to_image()
    cv3d.io.write_image("/tmp/test.png", img, 9)

    render.scene.camera.look_at([0, 0, 0], [-10, 0, 0], [0, 0, 1])
    img = render.render_to_image()
    cv3d.io.write_image("/tmp/test2.png", img, 9)


if __name__ == "__main__":
    main()