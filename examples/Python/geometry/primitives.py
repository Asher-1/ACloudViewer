# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/polyline.py

import numpy as np
import cloudViewer as cv3d


def primitives_generator(only_show_union=True):
    plane = cv3d.geometry.ccPlane(width=2, height=4)
    plane.paint_uniform_color((0.5, 1, 0))
    plane.compute_vertex_normals()
    if not only_show_union:
        yield "Plane", [plane]

    box = cv3d.geometry.ccBox(dims=[2, 2, 2])
    box.set_color([0, 0, 1])
    box.compute_vertex_normals()
    if not only_show_union:
        yield "Box", [box]

    sphere = cv3d.geometry.ccSphere(radius=2, precision=96)
    sphere.set_color((1, 0, 0))
    sphere.compute_vertex_normals()
    if not only_show_union:
        yield "Sphere", [sphere]

    torus = cv3d.geometry.ccTorus(inside_radius=1, outside_radius=1.5, rectangular_section=False,
                                  angle_rad=2 * np.pi, rect_section_height=0, precision=96)
    torus.set_color([0.5, 0, 1])
    torus.compute_vertex_normals()
    if not only_show_union:
        yield "Torus", [torus]

    quadric = cv3d.geometry.ccQuadric(min_corner=(-1, -1), max_corner=(1, 1), equation=(1, 1, 1, 1, 1, 1))
    quadric.paint_uniform_color((0, 1, 0.5))
    quadric.compute_vertex_normals()
    if not only_show_union:
        yield "Quadric", [quadric]

    truncated_cone = cv3d.geometry.ccCone(bottom_radius=2, top_radius=1, height=4, x_off=0, y_off=0, precision=64)
    truncated_cone.set_color([1, 0, 1])
    truncated_cone.compute_vertex_normals()
    if not only_show_union:
        yield "TruncatedCone", [truncated_cone]

    cone = cv3d.geometry.ccCone(bottom_radius=2, top_radius=0, height=4, x_off=0, y_off=0, precision=64)
    cone.set_color([1, 0, 1])
    cone.compute_vertex_normals()
    if not only_show_union:
        yield "Cone", [cone]

    cylinder = cv3d.geometry.ccCylinder(radius=2, height=4, precision=128)
    cylinder.paint_uniform_color((0, 1, 0))
    cylinder.compute_vertex_normals()
    if not only_show_union:
        yield "Cylinder", [cylinder]

    d = 4
    union = cv3d.geometry.ccMesh()
    union.create_internal_cloud()
    union += plane.translate((-d, 0, 0))
    union += box.translate((0, 0, 0))
    union += sphere.translate((0, -d, 0))
    union += torus.translate((-d, -d, 0))
    union += quadric.translate((d, 0, 0))
    union += truncated_cone.translate((d, -d, 0))
    union += cone.translate((-d, d, 0))
    union += cylinder.translate((d, d, 0))
    yield "Unions", [union]


if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)
    for name, primitives in primitives_generator(only_show_union=True):
        print(primitives)
        cv3d.visualization.draw_geometries(primitives)
