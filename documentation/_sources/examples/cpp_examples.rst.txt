C++ Examples
============

This page provides a comprehensive overview of C++ examples available in ACloudViewer. All examples are located in the ``examples/Cpp/`` directory.

.. note::
   
   To build and run these examples:
   
   .. code-block:: bash
   
      cd ACloudViewer
      mkdir build && cd build
      cmake ..
      make -j$(nproc)
      
      # Run an example
      ./bin/examples/PointCloud
      ./bin/examples/Visualizer

Basic Examples
--------------

Point Cloud Operations
^^^^^^^^^^^^^^^^^^^^^^

**PointCloud.cpp** - Basic point cloud manipulation

Demonstrates fundamental point cloud operations:

- Loading and saving point clouds
- Accessing point coordinates and normals
- Computing bounding boxes
- Point cloud statistics

.. code-block:: bash

   # Run
   ./bin/examples/PointCloud [pointcloud.pcd]
   
   # Features demonstrated:
   # - Read/write point clouds
   # - Print point cloud information
   # - Access point coordinates and normals
   # - Compute min/max bounds

**Key Code Snippet:**

.. code-block:: cpp

   #include "CloudViewer.h"
   
   auto pcd = std::make_shared<ccPointCloud>();
   cloudViewer::io::ReadPointCloud("input.pcd", *pcd);
   
   // Get bounding box
   Eigen::Vector3d min_bound = pcd->GetMinBound();
   Eigen::Vector3d max_bound = pcd->GetMaxBound();
   
   // Access points
   for (unsigned int i = 0; i < pcd->size(); i++) {
       const CCVector3 &point = *pcd->getPoint(i);
       // Process point...
   }

Triangle Mesh Operations
^^^^^^^^^^^^^^^^^^^^^^^^

**TriangleMesh.cpp** - Mesh creation and manipulation

Examples include:

- Creating primitive meshes (sphere, box, cylinder)
- Merging multiple meshes
- Computing mesh normals
- Mesh subdivision and simplification
- Mesh filtering

.. code-block:: bash

   # Run
   ./bin/examples/TriangleMesh sphere
   ./bin/examples/TriangleMesh merge mesh1.ply mesh2.ply
   ./bin/examples/TriangleMesh normal input.ply output.ply

**Key Operations:**

.. code-block:: cpp

   // Create sphere
   auto mesh = ccMesh::createSphere(1.0, 20);
   
   // Compute normals
   mesh->ComputeVertexNormals();
   
   // Merge meshes
   *mesh1 += *mesh2;
   
   // Paint mesh
   PaintMesh(*mesh, Eigen::Vector3d(1.0, 0.0, 0.0));

Visualization
-------------

Basic Visualization
^^^^^^^^^^^^^^^^^^^

**Visualizer.cpp** - Comprehensive visualization examples

Supports multiple visualization modes:

- Mesh visualization
- Point cloud visualization with colors
- Image and depth image display
- RGBD image pairs
- Interactive mesh editing
- Animation playback

.. code-block:: bash

   # Visualize mesh
   ./bin/examples/Visualizer mesh model.ply
   
   # Visualize point cloud with rainbow colors
   ./bin/examples/Visualizer rainbow cloud.pcd
   
   # Visualize RGBD image pair
   ./bin/examples/Visualizer rgbd color.png depth.png

**Supported Options:**

- ``mesh`` - Display 3D mesh
- ``pointcloud`` - Display point cloud
- ``rainbow`` - Point cloud with rainbow colors
- ``image`` - Display 2D image
- ``depth`` - Display depth image
- ``rgbd`` - Display RGBD pair
- ``editing`` - Interactive mesh editing
- ``animation`` - Play animation with trajectory

Multiple Windows
^^^^^^^^^^^^^^^^

**MultipleWindows.cpp** - Multi-window visualization

Demonstrates:

- Creating multiple visualization windows
- Synchronizing camera views
- Independent window controls
- Custom callback functions

.. code-block:: cpp

   // Create multiple windows
   auto vis1 = std::make_shared<visualization::Visualizer>();
   auto vis2 = std::make_shared<visualization::Visualizer>();
   
   vis1->CreateVisualizerWindow("Window 1", 640, 480);
   vis2->CreateVisualizerWindow("Window 2", 640, 480);
   
   // Add geometries
   vis1->AddGeometry(pcd1);
   vis2->AddGeometry(pcd2);

Offscreen Rendering
^^^^^^^^^^^^^^^^^^^

**OffscreenRendering.cpp** - Render without GUI

Useful for:

- Batch rendering
- Server-side visualization
- Automated image generation
- CI/CD pipelines

.. code-block:: cpp

   // Create offscreen renderer
   visualization::rendering::OffscreenRenderer renderer(640, 480);
   
   // Add geometry and render
   renderer.GetScene()->AddGeometry("mesh", mesh_ptr, material);
   auto image = renderer.RenderToImage();
   
   // Save rendered image
   io::WriteImage("rendered.png", *image);

Geometry Processing
-------------------

Voxelization
^^^^^^^^^^^^

**Voxelization.cpp** - Voxel-based operations

Examples:

- Voxel downsampling
- Voxel grid creation
- Occupancy grid
- Distance field computation

.. code-block:: cpp

   // Downsample point cloud
   auto downsampled = pcd->VoxelDownSample(voxel_size);
   
   // Create voxel grid
   auto voxel_grid = geometry::VoxelGrid::CreateFromPointCloud(
       *pcd, voxel_size);

Octree Operations
^^^^^^^^^^^^^^^^^

**Octree.cpp** - Spatial indexing with octrees

Features:

- Octree construction
- Nearest neighbor search
- Radius search
- Octree traversal
- Voxel location queries

.. code-block:: cpp

   // Create octree from point cloud
   auto octree = geometry::Octree::CreateFromPointCloud(*pcd, max_depth);
   
   // Find nearest neighbors
   auto [indices, distances] = octree->SearchKnn(query_point, k);
   
   // Radius search
   auto results = octree->SearchRadius(query_point, radius);

Half-Edge Triangle Mesh
^^^^^^^^^^^^^^^^^^^^^^^

**HalfEdgeTriangleMesh.cpp** - Advanced mesh representation

Topology operations:

- Half-edge data structure
- Mesh connectivity queries
- Edge operations
- Manifold checks

.. code-block:: cpp

   // Convert to half-edge mesh
   auto half_edge_mesh = geometry::HalfEdgeTriangleMesh::CreateFromTriangleMesh(*mesh);
   
   // Query connectivity
   auto adjacent_vertices = half_edge_mesh->GetAdjacentVertices(vertex_id);
   auto boundary_edges = half_edge_mesh->GetBoundaryEdges();

Registration
------------

ICP Registration
^^^^^^^^^^^^^^^^

**GeneralizedICP.cpp** - Iterative Closest Point

Point-to-point and point-to-plane ICP:

.. code-block:: cpp

   // Point-to-point ICP
   auto result = registration::RegistrationICP(
       source, target, max_distance,
       Eigen::Matrix4d::Identity(),
       registration::TransformationEstimationPointToPoint());
   
   // Point-to-plane ICP
   auto result = registration::RegistrationICP(
       source, target, max_distance,
       Eigen::Matrix4d::Identity(),
       registration::TransformationEstimationPointToPlane());

Colored ICP
^^^^^^^^^^^

**RegistrationColoredICP.cpp** - Color-enhanced registration

Uses both geometry and color information:

.. code-block:: cpp

   auto result = registration::RegistrationColoredICP(
       source, target, max_distance,
       Eigen::Matrix4d::Identity(),
       registration::ICPConvergenceCriteria());

Fast Global Registration
^^^^^^^^^^^^^^^^^^^^^^^^

**RegistrationFGR.cpp** - Feature-based registration

Fast global registration without initial alignment:

.. code-block:: cpp

   // Compute FPFH features
   auto source_fpfh = registration::ComputeFPFHFeature(
       *source, search_param);
   auto target_fpfh = registration::ComputeFPFHFeature(
       *target, search_param);
   
   // Fast global registration
   auto result = registration::FastGlobalRegistration(
       *source, *target, *source_fpfh, *target_fpfh, option);

RANSAC Registration
^^^^^^^^^^^^^^^^^^^

**RegistrationRANSAC.cpp** - Robust correspondence-based registration

.. code-block:: cpp

   // Find correspondences
   auto correspondences = registration::CorrespondencesFromFeatures(
       *source_fpfh, *target_fpfh);
   
   // RANSAC registration
   auto result = registration::RegistrationRANSACBasedOnCorrespondence(
       *source, *target, correspondences, max_distance);

Doppler ICP
^^^^^^^^^^^

**RegistrationDopplerICP.cpp** - Doppler velocity-enhanced ICP

For dynamic scenes with velocity information:

.. code-block:: bash

   ./bin/examples/RegistrationDopplerICP source.pcd target.pcd

Tensor-based ICP
^^^^^^^^^^^^^^^^

**TICP.cpp** - GPU-accelerated tensor ICP

High-performance registration using tensors:

.. code-block:: cpp

   // Create tensor point clouds
   auto source_t = t::geometry::PointCloud::FromLegacy(*source);
   auto target_t = t::geometry::PointCloud::FromLegacy(*target);
   
   // Tensor ICP
   auto result = t::pipelines::registration::ICP(
       source_t, target_t, max_distance, init);

3D Reconstruction
-----------------

RGBD Integration
^^^^^^^^^^^^^^^^

**IntegrateRGBD.cpp** - RGBD frame integration

Integrate RGBD frames into TSDF volume:

.. code-block:: cpp

   // Create TSDF volume
   auto volume = integration::ScalableTSDFVolume(
       voxel_length, sdf_trunc, color_type);
   
   // Integrate RGBD images
   for (const auto& [color, depth, pose] : frames) {
       auto rgbd = geometry::RGBDImage::CreateFromColorAndDepth(
           color, depth, depth_scale, depth_trunc);
       volume.Integrate(*rgbd, intrinsic, pose);
   }
   
   // Extract mesh
   auto mesh = volume.ExtractTriangleMesh();

Tensor RGBD Integration
^^^^^^^^^^^^^^^^^^^^^^^^

**TIntegrateRGBD.cpp** - GPU-accelerated TSDF integration

.. code-block:: cpp

   // Create voxel block grid
   auto voxel_grid = t::geometry::TSDFVoxelGrid({
       {"tsdf", core::Dtype::Float32},
       {"weight", core::Dtype::UInt16},
       {"color", core::Dtype::UInt16}
   });
   
   // Integrate frames
   voxel_grid.Integrate(depth, color, intrinsic, extrinsic);

SLAM System
^^^^^^^^^^^

**OfflineSLAM.cpp** - Offline SLAM pipeline

Complete SLAM system for recorded data:

.. code-block:: bash

   ./bin/examples/OfflineSLAM config.json

**OnlineSLAMRGBD.cpp** - Real-time RGBD SLAM

Online SLAM with live camera feed:

.. code-block:: cpp

   // Initialize SLAM
   auto slam = pipelines::slam::SLAMSystem(config);
   
   // Process frames
   for (const auto& frame : camera_stream) {
       slam.ProcessFrame(frame.color, frame.depth);
       auto pose = slam.GetCurrentPose();
   }

SLAC Optimization
^^^^^^^^^^^^^^^^^

**SLAC.cpp** - Simultaneous Localization and Calibration

Joint optimization of poses and sensor calibration:

.. code-block:: bash

   ./bin/examples/SLAC config.json fragments/

Reconstruction System
^^^^^^^^^^^^^^^^^^^^^

**Reconstruction.cpp** - Complete reconstruction pipeline

End-to-end 3D reconstruction:

- RGBD odometry
- Fragment creation  
- Fragment registration
- Integration
- Mesh refinement

.. code-block:: bash

   ./bin/examples/Reconstruction config.json

Odometry and Tracking
---------------------

RGBD Odometry
^^^^^^^^^^^^^

**RGBDOdometry.cpp** - Frame-to-frame odometry

.. code-block:: cpp

   // Compute odometry between two RGBD frames
   auto [success, transform, info] = odometry::ComputeRGBDOdometry(
       rgbd_source, rgbd_target, intrinsic,
       Eigen::Matrix4d::Identity(), odometry_option);

Tensor Odometry
^^^^^^^^^^^^^^^

**TOdometryRGBD.cpp** - GPU-accelerated odometry

High-performance odometry using tensors:

.. code-block:: cpp

   auto odometry = t::pipelines::odometry::RGBDOdometry(
       intrinsic_t, device);
   
   auto result = odometry.ComputeOdometry(
       source_rgbd_t, target_rgbd_t);

Camera Pose Trajectory
^^^^^^^^^^^^^^^^^^^^^^

**CameraPoseTrajectory.cpp** - Trajectory I/O and manipulation

.. code-block:: cpp

   // Load trajectory
   auto trajectory = camera::PinholeCameraTrajectory();
   io::ReadPinholeCameraTrajectory("trajectory.log", trajectory);
   
   // Access poses
   for (const auto& param : trajectory.parameters_) {
       auto pose = param.extrinsic_;
       // Use pose...
   }

Keypoint Detection
------------------

ISS Keypoints
^^^^^^^^^^^^^

**ISSKeypoints.cpp** - Intrinsic Shape Signatures

Detect salient keypoints in point clouds:

.. code-block:: cpp

   // Detect ISS keypoints
   auto keypoints = keypoint::ISS3DKeypoint(
       *pcd, salient_radius, non_max_radius, 
       gamma_21, gamma_32);
   
   // Compute descriptors at keypoints
   auto fpfh = registration::ComputeFPFHFeature(
       *keypoints, search_param);

SIFT/SURF Integration
^^^^^^^^^^^^^^^^^^^^^

**Image.cpp** - 2D image features

Extract and match image features:

.. code-block:: cpp

   // Load image
   auto image = io::CreateImageFromFile("image.png");
   
   // Extract features (via OpenCV integration)
   // Match features
   // Compute homography/fundamental matrix

File I/O and Formats
--------------------

Point Cloud Formats
^^^^^^^^^^^^^^^^^^^

**PCDFileFormat.cpp** - PCD file handling

Read/write PCD files with various options:

.. code-block:: cpp

   // Read PCD
   auto pcd = io::CreatePointCloudFromFile("input.pcd");
   
   // Write PCD (ASCII)
   io::WritePointCloudOption option;
   option.write_ascii = true;
   io::WritePointCloud("output.pcd", *pcd, option);
   
   // Write PCD (binary compressed)
   option.write_ascii = false;
   option.compressed = true;
   io::WritePointCloud("output_compressed.pcd", *pcd, option);

Supported formats:

- **Point Clouds**: PCD, PLY, XYZ, PTS, LAS, LAZ, E57
- **Meshes**: PLY, STL, OBJ, OFF, GLTF, FBX
- **Images**: PNG, JPG, BMP, TIF
- **Depth**: PNG (16-bit), TIF

File System Operations
^^^^^^^^^^^^^^^^^^^^^^

**FileSystem.cpp** - File and directory operations

.. code-block:: cpp

   #include <CloudViewer.h>
   
   // List files in directory
   auto files = utility::filesystem::ListFilesInDirectory(
       "data/", "*.pcd");
   
   // Check file existence
   if (utility::filesystem::FileExists("config.json")) {
       // Process file
   }
   
   // Create directories
   utility::filesystem::MakeDirectoryHierarchy("output/meshes/");

File Dialogs
^^^^^^^^^^^^

**FileDialog.cpp** - GUI file selection

.. code-block:: cpp

   // Open file dialog
   auto filename = utility::filesystem::GetOpenFileName(
       "Select Point Cloud", "data/",
       "Point Cloud Files (*.pcd *.ply);;All Files (*.*)");

Advanced Topics
---------------

Tensor Operations
^^^^^^^^^^^^^^^^^

**TICPSequential.cpp** - Sequential tensor ICP

Process sequence of scans with GPU acceleration:

.. code-block:: cpp

   // Device selection
   core::Device device("CUDA:0");
   
   // Create tensor geometry
   auto pcd_t = t::geometry::PointCloud(device);
   
   // Sequential registration
   Eigen::Matrix4d cumulative_transform = Eigen::Matrix4d::Identity();
   for (size_t i = 1; i < pcds.size(); i++) {
       auto result = t::pipelines::registration::ICP(
           pcds[i-1], pcds[i], max_distance, init);
       cumulative_transform = result.transformation_ * cumulative_transform;
   }

Gaussian Splatting
^^^^^^^^^^^^^^^^^^

**GaussianSplat.cpp** - 3D Gaussian splatting

Novel view synthesis using 3D Gaussians:

.. code-block:: bash

   ./bin/examples/GaussianSplat scene.ply

WebRTC Visualization
^^^^^^^^^^^^^^^^^^^^

**DrawWebRTC.cpp** - Remote visualization

Stream 3D data over WebRTC:

.. code-block:: cpp

   // Create WebRTC visualizer
   auto vis = visualization::webrtc_server::WebRTCWindowSystem::GetInstance();
   
   // Add geometry
   vis->AddGeometry("pointcloud", pcd);
   
   // Start server
   vis->Start();

Azure Kinect Examples
^^^^^^^^^^^^^^^^^^^^^

**AzureKinectViewer.cpp** - Azure Kinect camera support

.. code-block:: bash

   ./bin/examples/AzureKinectViewer

**AzureKinectRecord.cpp** - Record Kinect streams

.. code-block:: bash

   ./bin/examples/AzureKinectRecord output.mkv

**AzureKinectMKVReader.cpp** - Playback recordings

.. code-block:: bash

   ./bin/examples/AzureKinectMKVReader recording.mkv

RealSense Examples
^^^^^^^^^^^^^^^^^^

**RealSenseRecorder.cpp** - Intel RealSense recording

.. code-block:: bash

   ./bin/examples/RealSenseRecorder output.bag

**RealSenseBagReader.cpp** - Playback RealSense recordings

.. code-block:: bash

   ./bin/examples/RealSenseBagReader recording.bag

**OnlineSLAMRealSense.cpp** - Real-time SLAM with RealSense

.. code-block:: bash

   ./bin/examples/OnlineSLAMRealSense

Utility Examples
----------------

OpenMP Parallelization
^^^^^^^^^^^^^^^^^^^^^^

**OpenMP.cpp** - Multi-threaded processing

.. code-block:: cpp

   #include <omp.h>
   
   // Set number of threads
   omp_set_num_threads(8);
   
   #pragma omp parallel for
   for (int i = 0; i < point_cloud->size(); i++) {
       // Process point in parallel
   }

Logging System
^^^^^^^^^^^^^^

**Log.cpp** - Logging and debugging

.. code-block:: cpp

   // Set verbosity level
   utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
   
   // Log messages
   utility::LogInfo("Information message");
   utility::LogWarning("Warning message");
   utility::LogError("Error message");
   utility::LogDebug("Debug message");

Program Options
^^^^^^^^^^^^^^^

**ProgramOptions.cpp** - Command-line argument parsing

.. code-block:: cpp

   // Check for option
   if (utility::ProgramOptionExists(argc, argv, "--verbose")) {
       utility::SetVerbosityLevel(utility::VerbosityLevel::Debug);
   }
   
   // Get option value
   std::string input = utility::GetProgramOptionAsString(
       argc, argv, "--input", "default.pcd");
   
   double threshold = utility::GetProgramOptionAsDouble(
       argc, argv, "--threshold", 0.05);

Flann Integration
^^^^^^^^^^^^^^^^^

**Flann.cpp** - Fast nearest neighbor search

.. code-block:: cpp

   #include <CloudViewer.h>
   
   // Create Flann index
   geometry::KDTreeFlann kdtree;
   kdtree.SetGeometry(*pcd);
   
   // KNN search
   std::vector<int> indices;
   std::vector<double> distances;
   kdtree.SearchKNN(query_point, k, indices, distances);
   
   // Radius search
   kdtree.SearchRadius(query_point, radius, indices, distances);

Building Examples
-----------------

Build All Examples
^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   cd ACloudViewer
   mkdir build && cd build
   cmake -DBUILD_EXAMPLES=ON ..
   make -j$(nproc)
   
   # Examples are in build/bin/examples/

Build Specific Example
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   make PointCloud        # Build PointCloud example
   make Visualizer        # Build Visualizer example
   make RegistrationICP   # Build ICP example

Run Example with Test Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Download test data
   cd ACloudViewer
   python3 examples/test_data/download_test_data.py
   
   # Run with test data
   ./build/bin/examples/PointCloud examples/test_data/fragment.pcd
   ./build/bin/examples/Visualizer mesh examples/test_data/monkey.ply

Creating Custom Examples
-------------------------

Template Structure
^^^^^^^^^^^^^^^^^^

Use this template for new examples:

.. code-block:: cpp

   // ----------------------------------------------------------------------------
   // -                        CloudViewer: www.cloudViewer.org                  -
   // ----------------------------------------------------------------------------
   // Copyright (c) 2018-2024 www.cloudViewer.org
   // SPDX-License-Identifier: MIT
   // ----------------------------------------------------------------------------
   
   #include <iostream>
   #include "CloudViewer.h"
   
   void PrintHelp() {
       using namespace cloudViewer;
       PrintCloudViewerVersion();
       utility::LogInfo("Usage:");
       utility::LogInfo("    > MyExample [options]");
   }
   
   int main(int argc, char *argv[]) {
       using namespace cloudViewer;
       
       utility::SetVerbosityLevel(utility::VerbosityLevel::Info);
       
       if (argc < 2 || 
           utility::ProgramOptionExists(argc, argv, "--help")) {
           PrintHelp();
           return 1;
       }
       
       // Your code here
       
       return 0;
   }

Add to CMakeLists.txt
^^^^^^^^^^^^^^^^^^^^^

Add your example to ``examples/Cpp/CMakeLists.txt``:

.. code-block:: cmake

   cv3d_add_example(MyExample SRCS MyExample.cpp)
   cv3d_link_3rdparty_libraries(MyExample)

Resources
---------

- **Source Code**: `examples/Cpp/ <https://github.com/Asher-1/ACloudViewer/tree/main/examples/Cpp>`_
- **Python Examples**: :doc:`python_examples`
- **C++ API Reference**: :doc:`/cpp_api`
- **Tutorials**: :doc:`/tutorial/index`
- **Build Guide**: :doc:`/getting_started/build_from_source`

Contributing Examples
---------------------

We welcome new examples! See :doc:`/developer/contributing` for guidelines.

**Good examples should:**

- Demonstrate a specific feature or workflow
- Include clear comments
- Handle errors gracefully
- Provide usage instructions
- Use test data when possible
- Follow code style guidelines

Submit your examples via pull request on `GitHub <https://github.com/Asher-1/ACloudViewer/pulls>`_.
