# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Advanced/load_save_viewpoint.py

import numpy as np
import cloudViewer as cv3d


def save_view_point(pcd, filename):
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    cv3d.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = cv3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = cv3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pcd = cv3d.io.read_point_cloud("../../TestData/fragment.pcd")
    save_view_point(pcd, "viewpoint.json")
    load_view_point(pcd, "viewpoint.json")
