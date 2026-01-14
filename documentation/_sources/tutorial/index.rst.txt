Tutorial
========

.. toctree::
   :maxdepth: 1
   :caption: Basic Tutorials

   geometry/index
   visualization/index
   pipelines/index
   reconstruction/index
   ml/index
   advanced/index

.. toctree::
   :maxdepth: 1
   :caption: Core (Tensor) Tutorials

   core/tensor
   core/hashmap

.. toctree::
   :maxdepth: 1
   :caption: Tensor Geometry & Pipelines

   t_geometry/pointcloud
   t_pipelines/t_icp_registration
   t_pipelines/t_robust_kernel

Overview
--------

This tutorial provides step-by-step guides for using ACloudViewer.

Tutorial Structure
------------------

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Category
     - Description
   * - :doc:`geometry/index`
     - Working with 3D geometries (Point clouds, Meshes, etc.)
   * - :doc:`visualization/index`
     - Interactive 3D visualization
   * - :doc:`pipelines/index`
     - Processing pipelines (registration, integration)
   * - :doc:`reconstruction/index`
     - 3D reconstruction from images
   * - :doc:`ml/index`
     - Machine learning with 3D data
   * - :doc:`advanced/index`
     - Advanced topics

Getting Started
---------------

If you're new to ACloudViewer, start here:

1. :doc:`../getting_started/installation` - Install ACloudViewer
2. :doc:`../getting_started/quickstart` - Quick start guide
3. :doc:`geometry/pointcloud` - Your first point cloud
4. :doc:`visualization/visualization` - Basic visualization

Complete Examples
-----------------

All tutorials include complete, runnable code examples.

**Python Example:**

.. code-block:: python

   import cloudViewer as cv3d
   
   # Load and process
   pcd = cv3d.io.read_point_cloud("bunny.pcd")
   pcd_down = pcd.voxel_down_sample(0.05)
   
   # Visualize
   cv3d.visualization.draw([pcd_down], raw_mode=True)

**C++ Example:**

.. code-block:: cpp

   #include <cloudViewer/CloudViewer.h>
   
   int main() {
       auto pcd = cv::io::ReadPointCloud("bunny.pcd");
       auto pcd_down = pcd->VoxelDownSample(0.05);
       cv::visualization::DrawGeometries({pcd_down});
       return 0;
   }

See Also
--------

* :doc:`../cpp_api/index` - C++ API Reference
* :doc:`../getting_started/introduction` - Introduction

