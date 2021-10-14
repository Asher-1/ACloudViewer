import numpy as np
import cloudViewer as cv3d
import cloudViewer.visualization.gui as gui
import cloudViewer.visualization.rendering as rendering


def make_point_cloud(npts, center, radius):
    pts = np.random.uniform(-radius, radius, size=[npts, 3]) + center
    cloud = cv3d.geometry.ccPointCloud()
    cloud.set_points(cv3d.utility.Vector3dVector(pts))
    colors = np.random.uniform(0.0, 1.0, size=[npts, 3])
    cloud.set_colors(cv3d.utility.Vector3dVector(colors))
    return cloud


def high_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    vis = cv3d.visualization.O3DVisualizer("CloudViewer - 3D Text", 1024, 768)
    vis.show_settings = True
    vis.add_geometry("Points", points)
    for idx in range(0, points.size()):
        vis.add_3d_label(points.get_point(idx), "{}".format(idx))
    vis.reset_camera_to_default()

    app.add_window(vis)
    app.run()


def low_level():
    app = gui.Application.instance
    app.initialize()

    points = make_point_cloud(100, (0, 0, 0), 1.0)

    w = app.create_window("CloudViewer - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.CloudViewerScene(w.renderer)
    mat = rendering.Material()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling
    widget3d.scene.add_geometry("Points", points, mat)
    for idx in range(0, points.size()):
        widget3d.add_3d_label(points.get_point(idx), "{}".format(idx))
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()


if __name__ == "__main__":
    high_level()
    low_level()
