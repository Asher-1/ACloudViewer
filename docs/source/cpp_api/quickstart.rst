Quick Start
===========

This guide provides quick examples to get you started with ACloudViewer C++ API.

Prerequisites
-------------

Before you begin, ensure you have:

- C++ compiler with C++17 support (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15 or higher
- ACloudViewer installed or built from source

For installation instructions, see :ref:`compilation`.

Basic Examples
--------------

Loading a Point Cloud
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <ccPointCloud.h>
   #include <FileIOFilter.h>
   #include <iostream>

   int main() {
       // Load a point cloud from file
       ccPointCloud* cloud = FileIOFilter::LoadFromFile(
           "example.pcd",
           CC_SHIFT_MODE::AUTO,
           nullptr  // No dialog
       );
       
       if (cloud) {
           std::cout << "Loaded " << cloud->size() << " points" << std::endl;
           
           // Access point cloud properties
           std::cout << "Has colors: " << cloud->hasColors() << std::endl;
           std::cout << "Has normals: " << cloud->hasNormals() << std::endl;
           
           delete cloud;
       } else {
           std::cerr << "Failed to load point cloud" << std::endl;
           return 1;
       }
       
       return 0;
   }

Accessing Point Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <ccPointCloud.h>
   #include <CCGeom.h>

   void processPoints(ccPointCloud* cloud) {
       // Iterate through all points
       for (unsigned i = 0; i < cloud->size(); ++i) {
           // Get point coordinates
           const CCVector3* point = cloud->getPoint(i);
           double x = point->x;
           double y = point->y;
           double z = point->z;
           
           // Get point color if available
           if (cloud->hasColors()) {
               const ccColor::Rgb& color = cloud->getPointColor(i);
               // Use color.r, color.g, color.b
           }
           
           // Get normal if available
           if (cloud->hasNormals()) {
               const CCVector3& normal = cloud->getPointNormal(i);
               // Use normal.x, normal.y, normal.z
           }
       }
   }

Creating a New Point Cloud
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <ccPointCloud.h>

   ccPointCloud* createCloud() {
       // Create a new point cloud
       ccPointCloud* cloud = new ccPointCloud("MyCloud");
       
       // Reserve space for points
       cloud->reserve(1000);
       
       // Add points
       for (int i = 0; i < 1000; ++i) {
           CCVector3 point(
               static_cast<double>(i),
               static_cast<double>(i * 2),
               static_cast<double>(i * 3)
           );
           cloud->addPoint(point);
       }
       
       // Enable colors
       if (cloud->reserveTheRGBTable()) {
           for (unsigned i = 0; i < cloud->size(); ++i) {
               ccColor::Rgb color(
                   static_cast<ColorCompType>(i % 256),
                   static_cast<ColorCompType>((i * 2) % 256),
                   static_cast<ColorCompType>((i * 3) % 256)
               );
               cloud->setPointColor(i, color);
           }
           cloud->showColors(true);
       }
       
       return cloud;
   }

Saving a Point Cloud
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <ccPointCloud.h>
   #include <FileIOFilter.h>

   bool saveCloud(ccPointCloud* cloud, const QString& filename) {
       // Save to file
       CC_FILE_ERROR result = FileIOFilter::SaveToFile(
           cloud,
           filename,
           FileIOFilter::SaveParameters()
       );
       
       if (result == CC_FERR_NO_ERROR) {
           std::cout << "Cloud saved successfully" << std::endl;
           return true;
       } else {
           std::cerr << "Failed to save cloud: " << result << std::endl;
           return false;
       }
   }

Working with Meshes
~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <ccMesh.h>
   #include <ccPointCloud.h>

   ccMesh* createSimpleMesh() {
       // Create a point cloud for the mesh vertices
       ccPointCloud* vertices = new ccPointCloud("vertices");
       vertices->reserve(4);
       
       // Add 4 vertices (a square)
       vertices->addPoint(CCVector3(0, 0, 0));
       vertices->addPoint(CCVector3(1, 0, 0));
       vertices->addPoint(CCVector3(1, 1, 0));
       vertices->addPoint(CCVector3(0, 1, 0));
       
       // Create mesh
       ccMesh* mesh = new ccMesh(vertices);
       mesh->addTriangle(0, 1, 2);
       mesh->addTriangle(0, 2, 3);
       
       // Compute normals
       mesh->computeNormals();
       
       return mesh;
   }

Using PCL Algorithms
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <PCLEngine.h>
   #include <ccPointCloud.h>

   void applyPCLFilter(ccPointCloud* cloud) {
       // Apply statistical outlier removal
       PCLEngine engine;
       
       // Configure filter parameters
       engine.setParameter("meanK", 50);
       engine.setParameter("stddevMulThresh", 1.0);
       
       // Apply filter
       ccPointCloud* filtered = engine.applyStatisticalOutlierRemoval(cloud);
       
       if (filtered) {
           std::cout << "Filtered cloud has " << filtered->size() 
                     << " points (from " << cloud->size() << ")" << std::endl;
           // Use filtered cloud...
           delete filtered;
       }
   }

Octree Operations
~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include <DgmOctree.h>
   #include <ccPointCloud.h>

   void computeOctree(ccPointCloud* cloud) {
       // Compute octree
       DgmOctree* octree = cloud->getOctree();
       if (!octree) {
           octree = cloud->computeOctree();
       }
       
       if (octree) {
           // Get octree level
           unsigned char level = octree->findBestLevelForAGivenCellNumber(
               cloud->size()
           );
           
           // Perform nearest neighbor search
           DgmOctree::NeighboursSet neighbors;
           octree->getPointsInSphericalNeighbourhood(
               CCVector3(0, 0, 0),  // Query point
               0.1,                 // Radius
               neighbors,
               level
           );
           
           std::cout << "Found " << neighbors.size() 
                     << " neighbors within radius" << std::endl;
       }
   }

CMake Integration
-----------------

Basic CMakeLists.txt
~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.15)
   project(MyACloudViewerApp)

   # Find ACloudViewer
   find_package(ACloudViewer REQUIRED)

   # Create executable
   add_executable(my_app main.cpp)

   # Link libraries
   target_link_libraries(my_app PRIVATE 
       ACloudViewer::Core
       ACloudViewer::IO
       ACloudViewer::PCLEngine
   )

Advanced CMakeLists.txt
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.15)
   project(AdvancedApp)

   set(CMAKE_CXX_STANDARD 17)
   set(CMAKE_CXX_STANDARD_REQUIRED ON)

   # Find ACloudViewer with components
   find_package(ACloudViewer REQUIRED COMPONENTS
       Core
       IO
       DB
       PCLEngine
       Reconstruction
   )

   # Find Qt5 (if using GUI)
   find_package(Qt5 COMPONENTS Core Widgets OpenGL REQUIRED)

   # Source files
   set(SOURCES
       main.cpp
       processor.cpp
       visualizer.cpp
   )

   # Create executable
   add_executable(advanced_app ${SOURCES})

   # Link libraries
   target_link_libraries(advanced_app PRIVATE
       ACloudViewer::Core
       ACloudViewer::IO
       ACloudViewer::DB
       ACloudViewer::PCLEngine
       Qt5::Core
       Qt5::Widgets
       Qt5::OpenGL
   )

   # Include directories
   target_include_directories(advanced_app PRIVATE
       ${CMAKE_CURRENT_SOURCE_DIR}/include
   )

Complete Example Program
-------------------------

Here's a complete example that loads a point cloud, applies filtering, and saves the result:

.. code-block:: cpp

   #include <ccPointCloud.h>
   #include <FileIOFilter.h>
   #include <DgmOctree.h>
   #include <iostream>

   int main(int argc, char** argv) {
       if (argc < 3) {
           std::cerr << "Usage: " << argv[0] 
                     << " <input.pcd> <output.pcd>" << std::endl;
           return 1;
       }

       // Load point cloud
       std::cout << "Loading " << argv[1] << "..." << std::endl;
       ccPointCloud* cloud = FileIOFilter::LoadFromFile(
           argv[1],
           CC_SHIFT_MODE::AUTO,
           nullptr
       );
       
       if (!cloud) {
           std::cerr << "Failed to load point cloud" << std::endl;
           return 1;
       }
       
       std::cout << "Loaded " << cloud->size() << " points" << std::endl;
       
       // Compute octree for spatial operations
       std::cout << "Computing octree..." << std::endl;
       DgmOctree* octree = cloud->computeOctree();
       if (!octree) {
           std::cerr << "Failed to compute octree" << std::endl;
           delete cloud;
           return 1;
       }
       
       // Apply some processing (example: compute normals)
       std::cout << "Processing..." << std::endl;
       // Your processing code here...
       
       // Save result
       std::cout << "Saving to " << argv[2] << "..." << std::endl;
       CC_FILE_ERROR result = FileIOFilter::SaveToFile(
           cloud,
           argv[2],
           FileIOFilter::SaveParameters()
       );
       
       if (result == CC_FERR_NO_ERROR) {
           std::cout << "Success!" << std::endl;
       } else {
           std::cerr << "Failed to save: " << result << std::endl;
       }
       
       delete cloud;
       return 0;
   }

Next Steps
----------

- Explore the `Full API Reference <../cpp_api/index.html>`_ for detailed documentation
- Check out more examples in ``examples/Cpp/`` directory
- Read the :doc:`overview` for understanding the architecture
- Learn about specific modules:
  - :doc:`../python_api/cloudViewer.io` for I/O operations
  - :doc:`../python_api/cloudViewer.geometry` for geometric operations
  - :doc:`../tutorial/index` for advanced tutorials

Building from Source
---------------------

To build ACloudViewer from source and use it in your projects, see:

- :ref:`compilation` - Complete compilation guide
- `GitHub Repository <https://github.com/Asher-1/ACloudViewer>`_

Support
-------

- `GitHub Issues <https://github.com/Asher-1/ACloudViewer/issues>`_ for bug reports
- `GitHub Discussions <https://github.com/Asher-1/ACloudViewer/discussions>`_ for questions
