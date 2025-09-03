# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

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
    pcd_data = cv3d.data.PCDPointCloud()
    pcd = cv3d.io.read_point_cloud(pcd_data.path)
    save_view_point(pcd, "viewpoint.json")
    load_view_point(pcd, "viewpoint.json")
