# cloudViewer: www.cloudViewer.org
# The MIT License (MIT)
# See license file or visit www.cloudViewer.org for details

# examples/Python/Basic/python_binding.py

import cloudViewer as cv3d


def example_import_function():
    pcd = cv3d.io.read_point_cloud("../../test_data/ICP/cloud_bin_0.pcd")
    print(pcd)


def example_help_function():
    help(cv3d)
    help(cv3d.geometry.ccPointCloud)
    help(cv3d.io.read_point_cloud)


if __name__ == "__main__":
    example_import_function()
    example_help_function()
