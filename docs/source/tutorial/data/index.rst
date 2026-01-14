.. _dataset:

Dataset
=======

ACloudViewer comes with a built-in dataset module for convenient access to commonly
used example datasets. These datasets will be downloaded automatically from the
internet.

.. code-block:: python

    import cloudViewer as cv3d

    if __name__ == "__main__":
        dataset = cv3d.data.EaglePointCloud()
        pcd = cv3d.io.read_point_cloud(dataset.path)
        cv3d.visualization.draw([pcd])

.. code-block:: cpp

    #include <string>
    #include <memory>
    #include "cloudViewer/CloudViewer.h"

    int main() {
        using namespace cloudViewer;

        data::EaglePointCloud dataset;
        auto pcd = io::ReadPointCloud(dataset.GetPath());
        visualization::Draw({pcd});

        return 0;
    }

- Datasets are downloaded and cached automatically. The default data root is
  ``~/cloudViewer_data``. Data will be downloaded to ``~/cloudViewer_data/download``
  and extracted to ``~/cloudViewer_data/extract``.
- Optionally, you can change the default data root. This can be done by setting
  the environment variable ``CLOUDVIEWER_DATA_ROOT`` or passing the ``data_root``
  argument when constructing a dataset object.

PointCloud
~~~~~~~~~~

PCDPointCloud
-------------

Colored point cloud of a living room from the Redwood dataset in PCD format.

.. code-block:: python

    dataset = cv3d.data.PCDPointCloud()
    pcd = cv3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::PCDPointCloud dataset;
    auto pcd = io::ReadPointCloud(dataset.GetPath());

PLYPointCloud
-------------

Colored point cloud of a living room from the Redwood dataset in PLY format.

.. code-block:: python

    dataset = cv3d.data.PLYPointCloud()
    pcd = cv3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::PLYPointCloud dataset;
    auto pcd = io::ReadPointCloud(dataset.GetPath());

EaglePointCloud
---------------

Eagle colored point cloud.

.. code-block:: python

    dataset = cv3d.data.EaglePointCloud()
    pcd = cv3d.io.read_point_cloud(dataset.path)

.. code-block:: cpp

    data::EaglePointCloud dataset;
    auto pcd = io::ReadPointCloud(dataset.GetPath());

LivingRoomPointClouds
---------------------

57 point clouds of binary PLY format from the Redwood RGB-D Dataset.

.. code-block:: python

    dataset = cv3d.data.LivingRoomPointClouds()
    pcds = []
    for pcd_path in dataset.paths:
        pcds.append(cv3d.io.read_point_cloud(pcd_path))

.. code-block:: cpp

    data::LivingRoomPointClouds dataset;
    std::vector<std::shared_ptr<geometry::PointCloud>> pcds;
    for (const std::string& pcd_path: dataset.GetPaths()) {
        pcds.push_back(io::ReadPointCloud(pcd_path));
    }

OfficePointClouds
-----------------

53 point clouds of binary PLY format from Redwood RGB-D Dataset.

.. code-block:: python

    dataset = cv3d.data.OfficePointClouds()
    pcds = []
    for pcd_path in dataset.paths:
        pcds.append(cv3d.io.read_point_cloud(pcd_path))

.. code-block:: cpp

    data::OfficePointClouds dataset;
    std::vector<std::shared_ptr<geometry::PointCloud>> pcds;
    for (const std::string& pcd_path: dataset.GetPaths()) {
        pcds.push_back(io::ReadPointCloud(pcd_path));
    }

TriangleMesh
~~~~~~~~~~~~

BunnyMesh
---------

The bunny triangle mesh from Stanford in PLY format.

.. code-block:: python

    dataset = cv3d.data.BunnyMesh()
    mesh = cv3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::BunnyMesh dataset;
    auto mesh = io::ReadTriangleMesh(dataset.GetPath());

ArmadilloMesh
-------------

The armadillo mesh from Stanford in PLY format.

.. code-block:: python

    dataset = cv3d.data.ArmadilloMesh()
    mesh = cv3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::ArmadilloMesh dataset;
    auto mesh = io::ReadTriangleMesh(dataset.GetPath());

KnotMesh
--------

A 3D Mobius knot mesh in PLY format.

.. code-block:: python

    dataset = cv3d.data.KnotMesh()
    mesh = cv3d.io.read_triangle_mesh(dataset.path)

.. code-block:: cpp

    data::KnotMesh dataset;
    auto mesh = io::ReadTriangleMesh(dataset.GetPath());

RGBDImage
~~~~~~~~~

SampleRedwoodRGBDImages
-----------------------

Sample set of 5 color images, 5 depth images from the Redwood RGBD
living-room1 dataset. It also contains a camera trajectory log, a camera
odometry log, an rgbd match file, and a point cloud reconstruction obtained from
TSDF.

.. code-block:: python

    dataset = cv3d.data.SampleRedwoodRGBDImages()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        color_raw = cv3d.io.read_image(dataset.color_paths[i])
        depth_raw = cv3d.io.read_image(dataset.depth_paths[i])
        rgbd_image = cv3d.geometry.RGBDImage.create_from_color_and_depth(
                                                   color_raw, depth_raw)
        rgbd_images.append(rgbd_image)

    pcd = cv3d.io.read_point_cloud(dataset.reconstruction_path)

Demo
~~~~

DemoICPPointClouds
------------------

3 point cloud fragments of binary PCD format, from living-room1 scene of Redwood
RGB-D dataset. This data is used for ICP demo.

.. code-block:: python

    dataset = cv3d.data.DemoICPPointClouds()
    pcd0 = cv3d.io.read_point_cloud(dataset.paths[0])
    pcd1 = cv3d.io.read_point_cloud(dataset.paths[1])
    pcd2 = cv3d.io.read_point_cloud(dataset.paths[2])

.. code-block:: cpp

    data::DemoICPPointClouds dataset;
    auto pcd0 = io::ReadPointCloud(dataset.GetPaths()[0]);
    auto pcd1 = io::ReadPointCloud(dataset.GetPaths()[1]);
    auto pcd2 = io::ReadPointCloud(dataset.GetPaths()[2]);

DemoColoredICPPointClouds
-------------------------

2 point cloud fragments of binary PCD format, from apartment scene of Redwood
RGB-D dataset. This data is used for Colored-ICP demo.

.. code-block:: python

    dataset = cv3d.data.DemoColoredICPPointClouds()
    pcd0 = cv3d.io.read_point_cloud(dataset.paths[0])
    pcd1 = cv3d.io.read_point_cloud(dataset.paths[1])

.. code-block:: cpp

    data::DemoColoredICPPointClouds dataset;
    auto pcd0 = io::ReadPointCloud(dataset.GetPaths()[0]);
    auto pcd1 = io::ReadPointCloud(dataset.GetPaths()[1]);

For a complete list of available datasets, see :doc:`../../python_api/cloudViewer.data`.
