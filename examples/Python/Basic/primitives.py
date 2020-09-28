# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/polyline.py

import numpy as np
import cloudViewer as cv3d


def primitives_generator():
    cone = cv3d.geometry.ccCone(bottom_radius=2, top_radius=1, height=4, x_off=0, y_off=0)
    cone.set_color([1, 0, 1])
    cone.compute_vertex_normals()
    yield "Cone", [cone]

    cylinder = cv3d.geometry.ccCylinder(radius=2, height=4)
    cylinder.compute_vertex_normals()
    yield "Cylinder", [cylinder]

    sphere = cv3d.geometry.ccSphere(radius=2, precision=96)
    sphere.compute_vertex_normals()
    yield "Sphere", [sphere]

    box = cv3d.geometry.ccBox(dims=[2, 2, 2])
    box.compute_vertex_normals()
    yield "Box", [box]

    plane = cv3d.geometry.ccPlane(width=2, height=4)
    plane.compute_vertex_normals()
    yield "Plane", [plane]

    torus = cv3d.geometry.ccTorus(inside_radius=1, outside_radius=1.5, rectangular_section=False, angle_rad=2*np.pi,
                                  rect_section_height=0, precision=96)
    torus.compute_vertex_normals()
    yield "Torus", [torus]


if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)
    for name, primitives in primitives_generator():
        print(primitives)
        cv3d.visualization.draw_geometries(primitives)
