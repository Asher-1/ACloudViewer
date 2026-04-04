Quick Start
===========

This guide will help you get started with ACloudViewer in 5 minutes.

Python Quick Start
------------------

Basic Point Cloud Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cloudViewer as cv3d
   import numpy as np

   # Create a point cloud
   pcd = cv3d.geometry.ccPointCloud()
   points = np.random.rand(1000, 3)
   pcd.points = cv3d.utility.Vector3dVector(points)

   # Visualize
   cv3d.visualization.draw([pcd], raw_mode=True)

Load and Process Point Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cloudViewer as cv3d

   # Load point cloud
   pcd = cv3d.io.read_point_cloud("bunny.pcd")
   print(f"Loaded {len(pcd.points)} points")

   # Downsample
   pcd_down = pcd.voxel_down_sample(voxel_size=0.05)
   print(f"Downsampled to {len(pcd_down.points)} points")

   # Estimate normals
   pcd_down.estimate_normals(
       search_param=cv3d.geometry.KDTreeSearchParamHybrid(
           radius=0.1, max_nn=30))

   # Visualize
   cv3d.visualization.draw([pcd_down], raw_mode=True)

Point Cloud Registration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cloudViewer as cv3d

   # Load source and target
   source = cv3d.io.read_point_cloud("source.pcd")
   target = cv3d.io.read_point_cloud("target.pcd")

   # Downsample
   source_down = source.voxel_down_sample(0.05)
   target_down = target.voxel_down_sample(0.05)

   # Estimate normals
   source_down.estimate_normals(
       cv3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
   target_down.estimate_normals(
       cv3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

   # ICP registration
   threshold = 0.02
   reg_p2p = cv3d.pipelines.registration.registration_icp(
       source_down, target_down, threshold,
       cv3d.geometry.Tranformation.identity(4),
       cv3d.pipelines.registration.TransformationEstimationPointToPoint())

   print(f"Fitness: {reg_p2p.fitness}")
   print(f"RMSE: {reg_p2p.inlier_rmse}")

   # Apply transformation
   source.transform(reg_p2p.transformation)

   # Visualize
   cv3d.visualization.draw([source, target], raw_mode=True)

Surface Reconstruction
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cloudViewer as cv3d

   # Load point cloud
   pcd = cv3d.io.read_point_cloud("bunny.pcd")

   # Estimate normals (required for Poisson)
   pcd.estimate_normals()

   # Poisson reconstruction
   mesh, densities = cv3d.geometry.ccMesh.create_from_point_cloud_poisson(
       pcd, depth=9)

   # Remove low-density vertices
   vertices_to_remove = densities < np.quantile(densities, 0.01)
   mesh.remove_vertices_by_mask(vertices_to_remove)

   # Visualize
   cv3d.visualization.draw([mesh], raw_mode=True)

C++ Quick Start
---------------

Basic Example
~~~~~~~~~~~~~

.. code-block:: cpp

   #include <cloudViewer/CloudViewer.h>
   #include <iostream>

   int main() {
       // Create point cloud
       auto pcd = std::make_shared<cv::geometry::PointCloud>();
       
       // Add some points
       for (int i = 0; i < 1000; i++) {
           pcd->points_.push_back(
               Eigen::Vector3d(rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX),
                              rand() / double(RAND_MAX)));
       }

       // Visualize
       cv::visualization::DrawGeometries({pcd});

       return 0;
   }

Load and Process
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <cloudViewer/CloudViewer.h>

   int main() {
       // Load point cloud
       auto pcd = cv::io::ReadPointCloud("bunny.pcd");
       
       // Downsample
       auto pcd_down = pcd->VoxelDownSample(0.05);
       
       // Estimate normals
       pcd_down->EstimateNormals(
           cv::geometry::KDTreeSearchParamHybrid(0.1, 30));
       
       // Visualize
       cv::visualization::DrawGeometries({pcd_down});
       
       return 0;
   }

Desktop Application
-------------------

Launch the Application
~~~~~~~~~~~~~~~~~~~~~~

**Linux/macOS:**

.. code-block:: bash

   ./ACloudViewer

**Windows:**

Double-click the ACloudViewer icon or run from command line:

.. code-block:: powershell

   ACloudViewer.exe

Basic Operations
~~~~~~~~~~~~~~~~

1. **Load Point Cloud**: File → Open → Select PCD/PLY file
2. **Visualization**: Use mouse to rotate (left-click), pan (right-click), zoom (scroll)
3. **Filtering**: Tools → Filter → Outlier Removal
4. **Registration**: Tools → Registration → ICP
5. **Reconstruction**: Tools → Reconstruction → Poisson

Common Workflows
----------------

Workflow 1: Point Cloud Cleanup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cloudViewer as cv3d

   # Load noisy point cloud
   pcd = cv3d.io.read_point_cloud("noisy.pcd")

   # Statistical outlier removal
   cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
   pcd_clean = pcd.select_by_index(ind)

   # Downsample
   pcd_down = pcd_clean.voxel_down_sample(voxel_size=0.02)

   # Save cleaned point cloud
   cv3d.io.write_point_cloud("cleaned.pcd", pcd_down)

Workflow 2: Mesh Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import cloudViewer as cv3d

   # Load point cloud
   pcd = cv3d.io.read_point_cloud("scan.pcd")

   # Estimate normals
   pcd.estimate_normals()
   pcd.orient_normals_consistent_tangent_plane(100)

   # Create mesh
   mesh, densities = cv3d.geometry.ccMesh.create_from_point_cloud_poisson(
       pcd, depth=9)

   # Simplify mesh
   mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)

   # Save mesh
   cv3d.io.write_triangle_mesh("output.obj", mesh_simplified)

Next Steps
----------

* :doc:`../tutorial/index` - Detailed tutorials
* :doc:`../python_api/cloudViewer.core` - Python API reference
* :doc:`../python_example/geometry/index` - More examples

