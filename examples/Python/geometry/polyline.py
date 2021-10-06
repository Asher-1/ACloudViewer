# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/polyline.py

import numpy as np
import cloudViewer as cv3d


def get_circle(cx=0., cy=0., r=2., plane="xy"):
    theta = np.arange(0, 2 * np.pi, 0.01)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = np.zeros(y.shape[0])
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    if plane.lower() in ["xy", "yx"]:
        cloud = np.concatenate((x, y, z), axis=1)
    elif plane.lower() in ["xz", "zx"]:
        cloud = np.concatenate((x, z, y), axis=1)
    elif plane.lower() in ["yz", "zy"]:
        cloud = np.concatenate((z, x, y), axis=1)
    else:
        assert False, "invalid plane parameters!"
    return cloud


def generate_pivots():
    circle1 = get_circle(cx=0., cy=0., r=2., plane="xy")
    cloud1 = cv3d.geometry.ccPointCloud(cv3d.utility.Vector3dVector(circle1))
    circle2 = get_circle(cx=0., cy=0., r=2., plane="xz")
    cloud2 = cv3d.geometry.ccPointCloud(cv3d.utility.Vector3dVector(circle2))
    circle3 = get_circle(cx=0., cy=0., r=2., plane="yz")
    cloud3 = cv3d.geometry.ccPointCloud(cv3d.utility.Vector3dVector(circle3))
    polyline1 = cv3d.geometry.ccPolyline(cloud1)
    polyline2 = cv3d.geometry.ccPolyline(cloud2)
    polyline3 = cv3d.geometry.ccPolyline(cloud3)
    polyline1.set_color([1, 0, 0])
    polyline2.set_color([0, 1, 0])
    polyline3.set_color([0, 0, 1])
    return polyline1, polyline2, polyline3


def generate_spring():
    z = np.arange(-50, 50, 0.01)  # z坐标范围-50~50
    x = 50 * np.cos(z)
    y = 50 * np.sin(z)
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    points = np.concatenate((x, y, z), axis=1)
    return [wrap_polyline(points, color=(0, 1, 0))]


def generate_funnel(start, end, steps=0.1):
    z = np.arange(start, end, steps)
    x = z * np.cos(z)
    y = z * np.sin(z)
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))
    z = z.reshape((-1, 1))
    points = np.concatenate((x, y, z), axis=1)
    return points


def wrap_polyline(points, color):
    cloud = cv3d.geometry.ccPointCloud(cv3d.utility.Vector3dVector(points))
    polyline = cv3d.geometry.ccPolyline(cloud=cloud)
    # polyline.add(cloud)
    polyline.set_color(color)
    return polyline


def generate_from_file():
    entity = cv3d.io.read_entity("../../test_data/polylines/polylines.bin")
    polylines = entity.filter_children(recursive=False, filter=cv3d.geometry.ccHObject.POLY_LINE)
    print(polylines)
    for poly in polylines:
        poly.set_color([0, 0, 1])
        poly.set_width(10)
    return polylines


def polylines_generator():
    yield "polylines", generate_from_file()
    polyline = wrap_polyline(generate_funnel(-50, 0, 0.1), color=(1, 0, 0)) + \
               wrap_polyline(generate_funnel(0, 50, 0.1), color=(0, 1, 0))
    polyline.set_closed(False)
    polyline.set_2d_mode(True)
    yield "Spline", [polyline]
    yield "Spring", generate_spring()
    yield "Pivots", generate_pivots()


if __name__ == "__main__":
    cv3d.utility.set_verbosity_level(cv3d.utility.Debug)
    for name, polys in polylines_generator():
        print("{}: line width {}, color {}".format(name, polys[0].get_width(), polys[0].get_color()))
        print("{}: isClosed {}, 2D_mode {}".format(name, polys[0].is_closed(), polys[0].is_2d_mode()))
        cv3d.visualization.draw_geometries(polys)

        vertex_list = []
        for poly in polys:
            vertex = poly.sample_points(True, 10., True)
            vertex.paint_uniform_color(poly.get_color())
            vertex_list.append(vertex)
        cv3d.visualization.draw_geometries(vertex_list)