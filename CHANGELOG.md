ACloudViewer Version History
============================

v3.9.2 (Asher) - 12/22/2024
----------------------

## ACloudViewer 3.9.2 Release Notes
We are excited to present ACloudViewer 3.9.2!

We welcome you to the 3.9.2 beta release of ACloudViewer. This release is full of exciting new features with a strong emphasis in real-time pipelines, but also full of bug fixes and usability improvements. The big highlights of this release are as follows:

- New features:

	- New menu entry: Save project
		- File > Save project (or CTRL+SHIFT+S)
		- Saves all entities in the DB as a bin file

  - Tools > Fit > circle
		- fits a 2D circle on a 3D point cloud (thanks to https://github.com/truebelief)
		- works also on cylinders
  
	- CI support (continuous integration)
		- now using Github action CI
		- platform (Macos, Windows, Linux)
		- build (debug, release)
		- 32/64 bits
  
- New plugins

	- New unified plugin to load LAS files (by Thomas Montaigu)
		- based on LASzip
		- should work on all platforms (Windows, Linux, macOS)
		- manages all versions of LAS files (1.0 to 1.4)
		- gives much more control over extended fields (Extra-bytes VLR) as well as custom mapping between
			the existing fields of a cloud and their destination in the LAS file

	- VoxFall: non-parametric volumetric change detection for rockfalls
		- computes volume differences between 2 meshes, with some visual representation
  
	- New plugin: q3DMASC
		- 3DMASC is an advanced plugin for 3D point cloud classification, that uses Multiple Attributes, Scales and Clouds.
		  It is possible to use it with the GUI but also to call it with the command line.
		- See https://lidar.univ-rennes.fr/en/3dmasc

	- New plugin: qTreeIso
		- a 3D graph-based individual-tree isolator (treeiso) from Terrestrial Laser Scanning point clouds
		- by Zhouxin Xi and Chris Hopkinson, Artemis Lab, Department of Geography & Environment, University of Lethbridge (Canada)

	- New plugin: qPythonRuntime
		- Early step attempt at allowing to use Python to automate some stuff in ACloudViewer.
    - See https://github.com/tmontaigu/CloudCompare-PythonRuntime
		- The [documentation](https://tmontaigu.github.io/CloudCompare-PythonRuntime/index.html) is hosted using GitHub pages.

	- New Python-based plugin: 3DFin (3D Forest Inventory)
		- automatic computation of tree parameters in terrestrial point clouds
		- accessible via the Python plugin (check the option to install it via the Windows installer)
		- see https://github.com/3DFin/3DFin
		- developed at the Centre of Wildfire Research of Swansea University (UK) in collaboration with the
			Research Institute of Biodiversity (CSIC, Spain) and the Department of Mining Exploitation of
			the University of Oviedo (Spain)

- Enhancements:
	* Command line mode
		- I/O plugins are now loaded in command line mode (i.e. the FARO, DP and PCD formats can now be used)
		- set the default PCD output file format: -PCD_OUTPUT_FORMAT {format}
			- format can be one of 'COMPRESSED_BINARY', 'BINARY' or 'ASCII'
			- default format is 'COMPRESSED_BINARY'
  - PCD:
		- Can now load PCL files with integer xyz coordinates (16 and 32 bits) as well as double (64 bits) coordinates
		- Can now load 'scan grids' corresponding to structured clouds (so as to compute robust normals for instance)
		- the (standard ?) 16bytes alignment for the various fields has been removed, so as to drastically reduce the memory consumption and the output file size!
    - PCD now supports loading more field types (8 bit signed and unsigned, 16 bit signed and unsigned, 32 bit unsigned, 64 bit floating point)
    - PCD files can now be loaded or saved with local characters. PCD files will also be saved as compressed files by default.
    - PCD format:
  		- a new dialog will appear when saving PCD file, to choose the output format (between compressed binary, binary and ASCII/text)
  		- this dialog can be hidden once and for all by clicking on the 'Yes to all' button
  		- the default output format can also be set via the command line (see above)
  - The Animation plugin now uses ffmppeg 6.1

### supported platform:
- Windows `x86/64`
- Linux `x86/64`
- MacOS `X64 && arm64 (M1 and M2)`

v3.9.1 (Asher) - 15/11/2024
----------------------

## ACloudViewer 3.9.1 Release Notes
We are excited to present ACloudViewer 3.9.1!

We welcome you to the 3.9.1 beta release of ACloudViewer. This release is full of exciting new features with a strong emphasis in real-time pipelines, but also full of bug fixes and usability improvements. The big highlights of this release are as follows:
- Linux auto ci support for ubuntu18.04-ubuntu22.04.
- Fully supported docker building.
- Update PCL version from 1.11.1 to 1.14.1.
- Update VTK version from 8.2.0 to 9.3.1.
- migrate from C++14 to C++17 support as default.
- Python 3.12 support

### supported platform:
- Windows `x86/64`
- Linux `x86/64`
- MacOS `arm64 (M1 and M2)`

v3.9.0 (Asher) - 05/04/2023
----------------------

## ACloudViewer 3.9.0 Release Notes
We are excited to present ACloudViewer 3.9.0!

We welcome you to the 3.9.0 beta release of ACloudViewer. This release is full of exciting new features with a strong emphasis in real-time pipelines, but also full of bug fixes and usability improvements. The big highlights of this release are as follows:
- New Lock rotation axis feature.
- New Animation display features supported.
- New scalar filed switch feature supported.
- New ply mesh texture_u and texture_v supported.
- New platform support: MacOS!

### supported platform:
- Windows `x86/64`
- Linux `x86/64`
- MacOS `arm64 (M1 and M2)`

v3.8.0 (Asher) - 10/10/2021
----------------------

## ACloudViewer 3.8.0 Release Notes
We are excited to present ACloudViewer 3.8.0!

We welcome you to the 3.8.0 release of ACloudViewer. This release is full of exciting new features with a strong emphasis in real-time pipelines, but also full of bug fixes and usability improvements. The big highlights of this release are as follows:
- New 3D reconstruction system heavily based on colmap.
- New real-time 3D reconstruction pipeline, featuring GPU and CPU support based on VoxelHashing.
- New real-time point cloud registration algorithm, featuring a high-performance version of Iterative Closest Point (ICP).
- New Neighbor Search module, introducing your favorite search algorithms such as KNN and RadiusSearch, with support for GPU and CPU devices through a common interface.
- New web visualizer, which enables users to access the advanced rendering and visualization features of CloudViewer in your favourite web environments (remote and locally!), including Jupyter notebooks, Jupyter lab, and standalone web applications.
- New 3D machine learning models and datasets, featuring PointRCNN for 3D object detection, SparseConvNets for point cloud semantic segmentation, and support for ScanNet and SunRGBD.
- Upgraded GUI module, providing improved and more versatile versions of existing widgets, and new ones: ImageWidget and ToggleSwitch.
- Upgraded build system, adding support for CUDA 11.

### Real-time 3D reconstruction
We introduce a new CUDA accelerated pipeline including RGBD odometry, frame-to-model tracking, and volumetric integration.
![Figure 1. Example of 3D reconstruction from an RGB-D sensor.](doc/images/real-time-3D-Reconstruction.png)
#### Odometry
We introduce the tensor based real-time RGBD Odometry pipeline. In addition to the legacy Hybrid and Intensity based methods, we support the popular point-to-plane method.
#### TSDFVoxelGrid
We further accelerate volumetric integration and introduce fast ray casting for rendering.
#### VoxelHashing
Based on the accelerated RGBD odometry and raycasting, we present the fully functional VoxelHashing system. It performs dense volumetric reconstruction with fast frame-to-model tracking. We present an easy-to-use GUI that also shows real-time interactable surface reconstruction.
#### SLAC
We have further enhanced our legacy offline reconstruction system by introducing the Simultaneous Localization and Calibration (SLAC) algorithm. This algorithm applies advanced dense multi-way registration along with non-rigid deformation to create highly-accurate reconstructions.

### Real-time point cloud registration
We present a high-performance implementation of ICP using CloudViewer’ Tensor library. This module is one of the first on leveraging the new Neighbor search module and the newly crafted parallel kernels. This implementation brings support for multi-scale ICP, which allows us to do iterations on different resolutions in order to accelerate convergence while keeping computation low.
![Figure 2. ICP registration of multiple point clouds from a driving dataset.](doc/images/ICP-registration.png)
### New Neighbor Search module
Neighbor search is at the core of many 3D algorithms. Therefore, it is critical to have access to a fast implementation able to execute a large number of queries in a fraction of a second. After months of development, the CloudViewer team is proud to present the new Neighbor Search module!

This module brings support for core search algorithms, such as KNN, Radius search, and Hybrid search. All these algorithms are provided with support for both CPU and GPU, through a common and easy-to-use interface. Write your code once and support multiple devices! Moreover, we have not sacrificed a single flop of computation, making this module one of the fastest neighbor search libraries ever created.

### Web visualizer
The need for visualizing complex 3D data in web environments has surged considerably in the past few years, in part thanks to the proliferation of sensors like LIDAR and RGBD cameras. New use cases, such as online dataset inspection and remote visualization are now an integral part of many tasks, requiring the crafting of ad-hoc tools, which often are cumbersome to use.
![Figure 3. Standalone visualization of a semantic segmentation model in a browser.](doc/images/web_visualizer.png)
In order to improve this situation, we introduce our new web-based visualization module, which enables 3D visualization from any browsers and any location. This module lets users run advanced rendering and visualization pipelines, both remote and locally through your web browser. All the power of CloudViewer’ rendering engine --including support for PBR materials, multiple lighting systems, 3D ML visualization, and many other features--, are now supported in your browser. This module also includes a Jupyter extension for interactive web-based visualization! This new feature allows you to run compute-intensive 3D processing in a dedicated server while visualizing the results remotely on any device through your browser.
![Figure 4. Visualization of a 3D model on a Jupyter notebook.](doc/images/jupyter_visualizer.png)

### Build System
Our pip packages now include support for CUDA 11.0, PyTorch 1.7.1, and TensorFlow 2.4.1 to enable RTX 3000 series devices. Please, notice that we provide custom PyTorch wheels for Linux to work around an incompatibility between CUDA 11, PyTorch, and extension modules such as CloudViewer-ML.

This release also brings new improved support for CUDA on Windows. Users can now build CUDA accelerated Python wheels for Windows. CloudViewer is now built with security options enabled by default.

We hope you find CloudViewer 0.13.0 exciting and useful. Happy coding!
Remember that you can reach out with questions, requests, or feedback through the following channels:

[e-mail](ludahai19@163.com)

[Documentation](https://asher-1.github.io/docs)

[Downloads](https://asher-1.github.io/downloads)


v3.7.0 (Asher) - 11/12/2020
----------------------

## ACloudViewer 3.7.0 Release Notes
We are excited to present ACloudViewer 3.7.0!

CloudViewer 0.3.7 introduces a brand new 3D Machine Learning module, nicknamed [CloudViewer-ML](https://github.com/Asher-1/CloudViewer-ML). CloudViewer-ML is an extension of your favorite library to bring support for 3D domain-specific operators, models, algorithms, and datasets. In a nutshell, users can now create new applications combining the power of 3D data and state-of-the-art neural networks! CloudViewer-ML is included in all the binary releases of CloudViewer 0.3.7.

CloudViewer-ML comes with support for [Pytorch +1.4](https://pytorch.org/) and [TensorFlow +2.2](https://www.tensorflow.org), the two most popular machine learning frameworks. The first iteration of this module features a 3D semantic segmentation toolset, including training and inference capabilities for [RandlaNet](https://arxiv.org/abs/1911.11236) and [KPConv](https://arxiv.org/abs/1904.08889). The toolset supports popular datasets such as [SemanticKITTI](http://semantic-kitti.org), [Semantic3D](http://www.semantic3d.net), 3D Semantic Parsing of Large-Scale Indoor Spaces S3DIS, [Toronto3D](https://arxiv.org/abs/2003.08284), [Paris-Lille-3D](https://npm3d.fr/paris-lille-3d) and [Electricity3D](https://asher-1.github.io). CloudViewer-ML also provides a new model zoo compatible with Pytorch and TensorFlow, so that users can enjoy state-of-the-art semantic segmentation models without hassles.

We have endowed the new CloudViewer-ML module with a new data viewer tool. Users can now inspect their datasets and model’s predictions in an intuitive and simple way. This visualization tool includes support for Pytorch and TensorFlow frameworks and is fully customizable due to its Pythonic nature.

This viewer has been built upon the new visualization API, integrating the new Rendering and GUI modules. Thanks to the new visualization API, users can perform advanced rendering, fully programmatically from Python and C++. Users can also create slick GUIs with a few lines of Python code. Check out how to do this here.

- The CloudViewer app has also been extended to include the following features:

	- Support for FBX and glTF2 assets
	- Full support for PBR models.
	- CloudViewer 0.3.7 includes for the first time support for Linux ARM (64-bit) platforms. This has been a long-time requested 	feature that finally made it into the release. You can now enjoy all CloudViewer features, including our new rendering and visualization pipelines in OpenGL-enabled ARM platform.

[Breaking] Please, notice that the API and the structure of CloudViewer have changed considerably after an intense refactoring process. You will need to update your code to use the new namespaces. Please, check the full changelog and the documentation for further information.

We hope you find CloudViewer 0.3.7 exciting and useful. Happy coding!

Remember that you can reach out with questions, requests, or feedback through the following channels:

[e-mail](ludahai19@163.com)

[Discourse forum](https://asher-1.github.io)

[Discord network](https://asher-1.github.io)

The ACloudViewer team

- Legend:
	- [Added]: Used to indicate the addition of new features
	- [Changed]: Updates of existing functionalities
	- [Deprecated]: Functionalities / features removed in future releases
	- [Removed]: Functionalities/features removed in this release
	- [Fixed]: For any bug fixes
	- [Breaking] This functionality breaks the previous API and you need to check your code

- Installation and project structure
	- [Added] fetch Filament with CMake FetchContent (#2085)
	- [Added] speeds up compilation by caching 3rdparty downloads (#2155)
	- [Added] Show STATIC_WINDOWS_RUNTIME in cmake summary when configuring for Windows (#2176)
	- [Added] Move master releases to new bucket for lifecycle management (#2453)
	- [Added] added missing license header in ml module py files (#2333)
	- [Added] add vis namespace (#2394)
	- [Added] Devel wheels for users (#2429)
	- [Added] Build Filament on ARM64 (#2415)
	- [Changed] cmake: pickup python from PYTHON_EXECUTABLE or from PATH (#1923)
	- [Changed] avoid deduplication of the -gencode option (#1960)
	- [Changed] do not link main library when building the tf ops lib because of (#1981)
	- [Changed] do not use system jsoncpp (#2005)
	- [Changed] update Eigen to use the GitLab commit id (#2030)
	- [Changed] update formatter: clang-format-10, yapf 0.30.0 (#2149)
	- [Changed] disable CMP0104 warning (#2175)
	- [Changed] Build examples iteratively on Windows CI (#2199)
	- [Changed] simplify filament build-from-source (#2303)
	- [Changed] set cmake minimum to 3.15 to support generator expressions in (#2392)
	- [Changed] cmake 3.18 required for windows (#2435)
	- [Changed] ubuntu 20.04 build filament from source CI (#2438)
	- [Fixed] turobojpeg windows static runtime (#1876)
	- [Fixed] fix auto & warning (as error) for clang 10 (#1924)
	- [Fixed] Fix Eigen warning when using CUDA (#1926)
	- [Fixed] fix bug in import_3rdparty_library for paths without trailing '/' (#2084)
	- [Fixed] Fix tbb build (#2311)
	- [Fixed] Fix for cmake externalproject_add patch_command (#2431)
	- [Breaking] restructure project directory and namespace (#1970)
	- [Breaking] reorg: opend3d::gui -> cloudViewer::visualization::gui (#1979)
	- [Breaking] change folder case (#1993)
	- [Breaking] Reorg: Added namespace 'rendering' for visualization/rendering (#2002)
	- [Breaking] remove voxel_pooling namespace (#2014)
	- [Breaking] reorg: remove hash_* namespaces (#2025)
	- [Breaking] Rename GLHelper namespace (#2024)
	- [Breaking] Removed visualization::gui::util namespace (#2013) 
	- [Breaking] lower case "cloudViewer/3rdparty" intall dir (#2083)
	- [Breaking] refactor pybind namespace (#2249)
	- [Breaking] renamed torch python namespaces (#2330)
	- [Breaking] Move geometry to cloudViewer::t (#2403)
	- [Breaking] move io to cloudViewer::t (#2406)

- CORE features and applications
	- [Added] Add cleanup flag in TriangleMesh::select_by_index (#1883)
	- [Added] Orient normals using propagation on MST of Riemannian graph (#1895)
	- [Added] PointCloudIO: UnitTests and Benchmarks (#1891)
	- [Added] expose UniformTSDFVolume's origin in Python API (#1762)
	- [Added] cloudViewer_show_and_abort_on_warning(Core) (#1959)
	- [Added] ml-module (#1958)
	- [Added] add make check-cpp-style, apply-cpp-style (#2016)
	- [Added] ml op test code and torch reduce_subarrays_sum op (#2050)
	- [Added] CUDA header as system header for CMake 3.16 (#2058)
	- [Added] scalar support to more binary ops (#2093)
	- [Added] Tensor api demo (#2067)
	- [Added] Initial tensor-based pointcloud (#2074)
	- [Added] support Tensor.item() in pybind (#2130)
	- [Added] MKL integration with tests (#2128)
	- [Added] Linear algebra module (#2103)
	- [Added] rpc visualization interface (#2090)
	- [Added] Pick the color for all voxels when creating a dense VoxelGrid. (#2150)
	- [Added] Assimp Base integration (#2132)
	- [Added] ISS(Intrinsic Shape Signature) Keypoint Detection Module (#1966)
	- [Added] ask assimp to build zlib (#2188)
	- [Added] initial tensor-based image class (#2161)
	- [Added] Enable CI on Ubuntu 20.04 (focal) with CUDA on GCE (#2308)
	- [Added] ARM CI (#2414)
	- [Added] initial support for tensor-based mesh (#2167)
	- [Added] pybind for tpoincloud (#2229)
	- [Added] Add pybind for Application::PostToMainThread, fix grammar error in comment (#2237)
	- [Added] Tensor for custom object data types (#2244)
	- [Added] Nearest Neighbor module (#2207)
	- [Added] torch wrapper for voxel pooling (#2256)
	- [Added] Support cuda memory cacher (#2212)
	- [Added] ml-contrib subsample library (#2254)
	- [Added] python binding of NN class (junha/nn pybind) (#2246)
	- [Added] contrib neighbor search for ML ops (#2270)
	- [Added] GCE GPU CI docker (PyTorch + cuDNN) (#2211)
	- [Added] Re-add multithreaded performance improvements to ClassIO (#2230) 
	- [Added] torch continuous conv wrappers (#2287)
	- [Added] Support Float32 and Float64 neighbor search (#2241)
	- [Added] Layer interface for voxel_pooling op (#2322)
	- [Added] Fast compression mode for png writing (issue #846) (#2325)
	- [Added] Added pybinds for scene, fixed bug with CloudViewerScene and LOD (#2323)
	- [Added] NanoFlann Parallel Search (#2305)
	- [Added] XYZI format IO with tgeometry::PointCloud (#2356)
	- [Added] Support CPU/GPU hashmap (#2226)
	- [Added] OpenBLAS/LAPACK and support for ARM (#2205)
	- [Added] Added max error threshold (#2411)
	- [Added] Add function to compute the volume of a watertight mesh (#2407)
	- [Added] Ray/line to axis-aligned bounding box intersections (#2358)
	- [Added] IO wrapper for geometry::PointCloud -> t::geometry::PointCloud (#2462)
	- [Added] Nacho/robust kernels (#2425)
	- [Changed] test_color_map.py: adjust rtol to allow enough FP tolerance for OPENMP reordering; add .request to all import urllib (#1897)
	- [Changed] Refactor CMake buildsystem (#1782)
	- [Changed] Refactor pointcloud tests (#1925)
	- [Changed] expose poisson rec threads param (#2035) 
	- [Changed] TensorList refactoring and comparison tensor ops (#2066)
	- [Changed] updated internal fields of conv layers to ease debugging (#2104)
	- [Changed] speeds up legacy pointcloud converter (#2216)
	- [Changed] Update TriangleMeshSimplification.cpp (#2192)
	- [Changed] object-oriented dtype (#2208)
	- [Changed] use pybind11's gil management (#2278)
	- [Changed] use link target torch_cpu for cpu builds (#2292)
	- [Changed] make 3rdparty_tbb depend on ext_tbb (#2297)
	- [Changed] Update camera.cpp (#2312)
	- [Changed] Delay cuSOLVER and cuBLAS init so exceptions are transferred to Python. (#2319)
	- [Changed] convert noncontiguous tensors instead of throwing an exception. (#2354)
	- [Changed] Refector some image tests failing on arm simulator (#2393)
	- [Changed] Ensure C++ and Python units tests always run (#2428)
	- [Removed] disable CreateFromPointCloudPoisson test for macos (#2054)
	- [Removed] Remove looking for X11 on macOS (#2334)
	- [Removed] Remove unused variable from SolveJacobianSystemAndObtainExtrinsicMatrix (#2398)
	- [Removed] Nacho/remove openmp guards (#2408)
	- [Fixed] fix coord frame origin bug (#2034)
	- [Fixed] fix utility::LogXX {} escape problem (#2072)
	- [Fixed] Release Python GIL for fast multithreaded IO (#1936)
	- [Fixed] PointToPlane and ColoredICP only require target normal vectors. Fix #2075 (#2118)
	- [Fixed] fixed radius search op for torch (#2101)
	- [Fixed] fix windows python dtype convert (#2277) 
	- [Fixed] list pytorch device correctly for pytest (#2304)
	- [Fixed] Fix handling of 4 channel images on PNG export (#2326)
	- [Fixed] Fix for "symbol already registered" (#2324)
	- [Fixed] Slice out-of-range (#2317)
	- [Fixed] fix NanoFlannIndexHolderBase mem leak (#2340)
	- [Fixed] fix c_str() temp address warning (#2336)
	- [Fixed] fix required_gradient propagation for the offset parameter (#2350)
	- [Fixed] -bugfix for float extents for torch cconv layer (#2361)
	- [Fixed] Fix nvidia download error (#2423)
	- [Fixed] fix filament default CLANG_DEFAULT_CXX (#2424)

- Rendering and visualization
	- [Added] App: add option to material combobox to restore to original values from file (#1873)
	- [Added] Add support for PNG with alpha channel (#1886)
	- [Added] GUI: implements image widget (#1881)
	- [Added] GUI: added threading and loading dialog to app (#1896)
	- [Added] Integrates point cloud I/O progress into app loading progress bar (#1913)
	- [Added] Adds menu option on macOS to make CloudViewer viewer default for file types (#2031)
	- [Added] Implements python bindings for gui namespace (#2042)
	- [Added] Added gui::TreeView widget (#2081)
	- [Added] Adds ability to set button padding (#2082)
	- [Added] gui::TreeView now supports arbitrary widgets for its cells (#2105)
	- [Added] Added FBX to CloudViewerViewer dialog (#2204)
	- [Added] GUI changes for CloudViewer-ML visualization (#2177)
	- [Added] Enable transparency for lit material (#2239)
	- [Added] Add shader for materials with transparency (#2258)
	- [Added] Unconditionally take base color from MATKEY_COLOR_DIFFUSE (#2265)
	- [Added] Added unlitGradient shader for colormaps and LUT (#2263)
	- [Added] Expose method to update vertex attributes (#2282)
	- [Added] Added ability to change downsample threshold in CloudViewerScene (#2349)
	- [Added] Faster Filament geometry creation for TPointCloud (sometimes up to 90%) (#2351)
	- [Added] Better algorithm for downsampling (#2355)
	- [Added] Add bounding-box-only mode for rotation. (#2371)
	- [Added] Add "__visualization_scalar" handling to FilamentScene::UpdateGeometry (#2376)
	- [Added] Enable python to render to image (#2413)
	- [Added] Added ability to set the preferred with of gui::NumberEdit (#2373)
	- [Added] Expose caching related functions in Python (#2409)
	- [Added] TPointCloud support for new Scene class (#2213)
	- [Added] Downsample point clouds with by using a different index array (#2318)
	- [Added] Added unlitSolidColor shader (#2352)
	- [Added] Added special name for TPointCloud rendering to create UV arrays from scalar on C++ side (#2363)
	- [Added] Add has_alpha to pybind for Material (#2383)
	- [Added] Add alpha to baseColor of shader (and simplify some shader calculations) (#2396)
	- [Changed] update_progress callbacks for ReadPointCloud and WritePointCloud (#1907)
	- [Changed] only update single geometry in Visualizer::AddGeometry and Visualizer::RemoveGeometry (#1945)
	- [Changed] Updated Info.plist file for app to claim it edits and is the default type for the file associations. Also adds .pcd as a supported file type. (#2001)
	- [Changed] overload draw_geometries (#1997)
	- [Changed] Visualization refactor: CloudViewerViewer and rendering::Scene (#2125)
	- [Changed] Disable MTL file saving for OBJ files with no UVs and textures (issue #1974) (#2164)
	- [Changed] Upgrade CloudViewer to use Filament 1.8.1 (#2165)
	- [Changed] Improve UI responsiveness when viewing large models (#2384)
	- [Changed] Force scene redraw when layout changes (#2412)
	- [Fixed] Fix window showing buffer from last resize when a window is moved on macOS (#2076)
	- [Fixed] Fixed crash when create an autosizing window with no child windows (which would autosize to (0, 0), which doesn't go over well) (#2098)
	- [Fixed] Fixed hidpi on Linux (#2133)
	- [Fixed] Tell macOS that Python using GUI on macOS works like an app (#2143)
	- [Fixed] Fix GUI to build with Windows (#2153)
	- [Fixed] Model loading and rendering (#2194)
	- [Fixed] Fix rendering anomalies with ao/rough/metal texture (#2243)
	- [Fixed] Fix vis-gui.py not being able to load any geometry (#2273)
	- [Fixed] fix screen rendering in offscreen mode (#2257)
	- [Fixed] Fix 'All' option in file dialog in vis-gui.py (#2274)
	- [Fixed] Fix left side of checkboxes in a treeview on the left not being clickable (#2301)
	- [Fixed] Fix checkboxes in treeview not always redrawing when clicked (#2314)
	- [Fixed] Fix crash on Abandoned FBX model (#2339)
	- [Fixed] Fixes for UI anomalies caused by responsiveness changes (#2397)
	- [Fixed] Force redraw when menu visibility changes (#2416)
	- [Fixed] Fix Scene::SetBackgroundColor() not working if ShowSkybox() not previously called (#2452)
	- [Fixed] Caching Related Bug Fixes (#2451)
	- [Fixed] Clean up Filament to avoid crashes. (#2348)
	- [Removed] Eliminate union in MaterialParameter as it was being used incorrectly (#1879)

- Improvements
	- Update VTK Rendering Engine from QVTKWidgets to QVTKOpenGLNativeWidgets, faster!
	- Update rendering window when call removeFromDB function.
	- Simplify installation.
	- Support CUDA SpeedUp now!
	- Support Linux and Mac Platform now!
	- Better support for High DPI screens (4K) on Windows, Linux and Mac platforms!


v3.6.0 (Asher) - 12/10/2020
----------------------

- New features
	- Edit > Clean > Voxel Sampling: to voxel sample point cloud with voxel size given by users.
	- Edit > Mesh > Convex Hull: to compute convex hull for point cloud.
	- Edit > Mesh > Poisson Reconstruction: to triangulation on point cloud given by users with poisson algorithm
	- Edit > Scalar Field > Import SF from file: to import scalar field data from file.
	- Edit > Scalar Field > Filter by lables: to filter point clouds by labels in scalar field mode imported from file or exported from semantic annotation tools.
	- New tool:
		- Tools > Segmentation > DBSCan Cluster
			- Cluster ccPointCloud using the RANSAC algorithm. Wrapper to Schnabel et al. library for automatic shape detection in point cloud, "Efficient RANSAC for Point-Cloud Shape Detection", Ruwen Schnabel, Roland Wahl and Reinhard Klein, in Computer Graphics Forum(June 2007), 26:2(214 - 226) http://cg.cs.uni-bonn.de/en/publications/paper-details/schnabel-2007-efficient/. 
			- Returns a list of ransac point labels and shape entity(ccGenericPrimitive).
		- Tools > Segmentation > Plane Segmentation
			- Segment ccPointCloud plane using the RANSAC algorithm.
			- Returns the plane model ax + by + cz + d = 0 and the indices of the plane inliers.
	- New Plugins:
		- qColorimetricSegmenter, The purpose of the plugin is to perform color segmentation.
			The available filters are RGB, HSV and scalar value filtering.
		- qHoughNormals, Normal Estimation in Unstructured Point Clouds with Hough transform.
		- qMPlane, MPlane is to perform normal distance measurements against a defined plane.
			- Fit a plane through a point cloud by selecting at minimum 3 reference points. A scalarfield for visualizing the normal distance is created and applied automatically.
			- Perform normal distance measurements against the plane.
			- Save measurements as a .csv file.
		- qMasonry, enables the segmentation of dense point clouds (principally from laser scanning) of masonry structures into their individual stones. The plugin contains two tools: one is for the automated segmentation of the point cloud into the wall's constitutive stones. The other one is to conduct this process manually either from scratch or (most commonly) to correct the errors of the automated tool (it's hard to create a perfect tool!).
			- qAutoSeg. The objective of this plugin is to automatically detect stones and mortar joints within a masonry wall, segmenting and saving the outcomes as distinct entities that can be later exploited independently. More information on the segmentation algorithm can be found in Valero et al.
			- qManualSeg. This plugin can be used to manually segment a point cloud of a masonry wall, or to correct and edit the segmentation of one that has been already subject to a previous segmentation process.

- Improvements
	- More efficient scalar field rendering strategies.
	- Update rendering window when call removeFromDB function.

- Changes
	- Reconstruct CVCoreLib, ECV_DB_LIB, ECV_IO_DB direction structure.
	- Add CVAppCommon, CVPluginAPI and CVPluginStub modules.
	- Reconstruct plugin Standard and IO module direction structure.
	- Remove Contribs directory and put some third parties into plugins itself.
	- add more python interfaces for custom geometry like primitives and some generic mesh and cloud geometry.

- Bug fixes:
	- Repair primitives normals showing mode in OpenGL mode for python interface.
	- Remove some dummy cmake steps and clearn cmakelist files.
	- Auto seg plugin or manual seg plugin: at log time, the log file should not be created in software directory. should be as follows:
		- APS_log_timestamp_random.txt for auto seg plugin
		- MSP_log_timestamp_random.txt for manual seg plugin


v3.5.0 (Asher) - 06/08/2020
----------------------

- New features
	- Headless rendering
	- Non-blocking visualization
	- Docker for cloudViewer
	- Refined build system
	- Fast global registration
	- Color map optimization
	- PyPi support
	- Changing Python package name: py3d -> cloudViewer
	- Many bug fixes
	- support jupyter interface in python mode
	- 3 methods to quickly change the coordinate system of one or several entities
		- Tools > Registration > Move bounding-box center to origin
		- Tools > Registration > Move bounding-box min corner to origin
		- Tools > Registration > Move bounding-box max corner to origin
	- Edit > Plane > Flip: to flip the selected planes
	- Edit > Plane > Compare: to compare the two selected planes (angle and relative distances)
	- Edit > Mesh > Flip triangles: to flip the triangles (vertices) in case they are defined in the wrong order
	- Tools > Distances > Cloud/Primitive Dist
		- Used to calculate distance to primitive shape (supports spheres, planes, cylinders, cones and boxes) rather than the mesh of that shape (more accurate results for spheres)
		planes are now optionally treated as bounded or unbounded.
		for cones this will not work with Snout mode cones.
	- New tool:
		- Edit > Normals > Export normals to SF(s) (or equivalently Edit > Scalar fields > Export normals to SF(s))
		- Command line argument: -NORMALS_TO_SFS (all dimensions are exported by default as 3 scalar fields)
	- Command line:
		- '-H_EXPORT_FMT' added to select format for hierarchy objects exported
		- The PCV tool can now be accessed via the command line:
			- Option '-PCV' (see https://www.cloudcompare.org/doc/wiki/index.php?title=Command_line_mode for sub-options)
			- Can be called on any number of clouds or meshes
			- (the tool was already accessible in V2.10, but in a very limited way)
		- RANSAC plugin support added, all parameters below are optional and can be added in any order, and will work on all clouds opened and already loaded when called
			- '-RANSAC' (main command)
				- EPSILON_ABSOLUTE - max distance to primitive
				- EPSILON_PERCENTAGE_OF_SCALE - max distance to primitive as a percentage of cloud scale 0.0-1.0 exclusive
				- BITMAP_EPSILON_PERCENTAGE_OF_SCALE - sampling resolution as a percentage of cloud scale 0.0-1.0 exclusive
				- BITMAP_EPSILON_ABSOLUTE - sampling resolution
				- SUPPORT_POINTS - min Support points per primitive
				- MAX_NORMAL_DEV - max normal deviation from the ideal shape normal vector [in Degrees]
				- PROBABILITY - probability that no better candidate was overlooked during sampling, the lower the better!
				- OUT_CLOUD_DIR - path to save detected shapes clouds to, current dir if unspecified
				- OUT_MESH_DIR - path to save detected shapes meshes to, current dir if unspecified
				- OUT_PAIR_DIR - path to save detected shapes clouds & meshes to, current dir if unspecified
				- OUT_GROUP_DIR - path to save all shapes and primitives to as a single file, current dir if unspecified
				- OUTPUT_INDIVIDUAL_SUBCLOUDS - output detected shapes clouds
				- OUTPUT_INDIVIDUAL_PRIMITIVES - output detected shapes meshes
				- OUTPUT_INDIVIDUAL_PAIRED_CLOUD_PRIMITIVE - output detected shapes clouds & meshes
				- OUTPUT_GROUPED - output all detected shapes clouds & meshes as single file
				- ENABLE_PRIMITIVE - each shape listed after this option will be searched for
					- Shapes are: PLANE, SPHERE, CYLINDER, CONE, TORUS
	    - '-NORMALS_TO_DIP': converts the loaded cloud normals to dip and dip direction (scalar fields)
	    - '-NORMALS_TO_SFS': converts the loaded cloud normals to 3 scalar fields (Nx, Ny and Nz)
		- '-REMOVE_RGB': to remove RGB colors (from all loaded entities, i.e. clouds or meshes)
		- '-REMOVE_Normals': to remove normals (from all loaded entities, i.e. clouds or meshes)
	    - The 1st Order Moment tool (Tools > Other > Compute geometric features) can now be accessed via
	    	the command line mode with option -MOMENT {kernel size}
	    	- Computes 1st order moment on all opened clouds and auto saved by default.
	    - The Feature tools (Tools > Other > Compute geometric features) can now be accessed via
    		the command line mode with option -FEATURE {type} {kernel size}
	    	- type is one of the following: SUM_OF_EIGENVALUES, OMNIVARIANCE, EIGENTROPY, ANISOTROPY, PLANARITY, LINEARITY, PCA1, PCA2, SURFACE_VARIATION, SPHERICITY, VERTICALITY, EIGENVALUE1, EIGENVALUE2, EIGENVALUE3.
	- 4 new default color scales:
		- Brown > Yellow
		- Yellow > Brown
		- Topo Landserf
		- High contrast

- Improvements
	- Better support for High DPI screens (4K) on Windows
	- Both the local and global bounding-box centers are now displayed in the cloud properties (if the cloud has been shifted)
	- The PoissonRecon plugin now relies on the PoissonRecon V12 library
		- new algorithm
		- option to set the final 'resolution' instead of the octree depth
	- Align (Point-pair based registration) tool
		- can now be used with several entities (both several aligned and several reference entities)
		- option to pick the center of sphere entities as registration point(CC will ask whether to use the sphere center or not when picking a point anywhere on a sphere entity)
	- All parameters should now be properly remembered from one call to the other (during the same session)
	- The current box/slice position can now be exported (resp. imported) to (resp. from) the clipboard via the 'Advanced' menu
	- Command line tool:
		- The C2M_DIST command (Cloud-to-Mesh distances) can now be called with 2 meshes as input. In this case the first mesh vertices are used as compared cloud.
		- New suboption for the -O -GLOBAL_SHIFT option: 'FIRST'
			To use the first encountered (non null) global shift for all loaded entities (must be defined for all entities nevertheless ;)
		- The CROP command will now remove the input cloud if it's totally 'cropped out' (instead of leaving the full original cloud loaded)
		- The 'FWF_O' command (to load LAS files with associated waveform data) now properly supports the '-GLOBAL_SHIFT' option
		- No more popup will appear when loading a raster file via the command line mode in SILENT mode (raster is never loaded as a textured quad, and invalid points are always ignored and not loaded)
		- One can now use the 'auto' value for the radius value input after -OCTREE_NORMALS to automatically guess the normal computation radius
	- Graphical segmentation:
		- points are now exclusively segmented inside/outside the frustum
	- Plugins:
		- plugins may now be enabled/disabled in the plugin info window
		- to take effect, ACloudViewer must be restarted
		- all plugins are still available on the command line
	- PCD now supports loading more field types (16 bit signed and unsigned, 32 bit unsigned, 64 bit floating point)
	- OBJ files:
		- we now correctly handle faces with more than 4 vertices! (they should be properly tessellated)
		- support of escaped lines ('\' at the end of the line)
		- now accepts MTL files with the 'Tf' keyword (well, CC just ignores it and doesn't complain about a wrong MTL file anymore ;)
		- enhanced progress report (thanks to https://gitlab.com/Epic_Wink)
	- M3C2:
		- the computation speed should be improved when using a small projection radius (smarter selection of the octree level)
	- LAS files:
		- the standard LAS Filter now handles the OVERLAP classification bit (for point format >= 6)
		- improved/fixed management of classification and classification flags
		- LAS offset (chosen at saving time) should be a little bit smarter:
			- CC will try to keep the previous one, or use the bounding-box min corner ONLY if the coordinates are too large
			- CC won't use the previous scale if it is too small for the current cloud
			- the 'optimal' scale is simpler (round values + the same for all dimensions)
		- The LAS 1.3/1.4 filter (Windows only) has been improved:
			- option to save any scalar field as extra bytes (and load extra bytes as scalar fields)
			- proper minor version setting
			- proper header size
			- using the latest version of LASlib with proper management of UTF8/UTF16 ('foreign') characters
	- ASCII files:
		- now allows mixed white space (spaces / tabs)
		- the ASCII load dialog option has now an option to load numerical values with a comma as digit separator ('use comma as decimal character' checkbox)
	- Unroll
		- ability to set the start and stop angles for the cone unrolling options
		- ability to unroll meshes
		- new unrolling mode: 'Straightened cone' (the previous one has been renamed 'Straightened cone (fixed radius)'). This new mode unrolls the cone as a cylinder but with a varying radius.
	- The 'Straightened cone' options are now using the real curvilinear abscissa (0 = cone apex)
	- Tools > Others > Compute geometric features
		- option to compute the 1st moment added
		- option to compute the 1st, 2nd and 3rd eigenvalues added
	- SBF files
		- format slightly updated to accommodate with scalar fields 'shift' (backward compatibility maintained)
		- format description is here: https://www.cloudcompare.org/doc/wiki/index.php?title=SBF

- Changes
	- Command line tool:
		- The `-FBX_EXPORT_FMT` command is now split. Use `-FBX -EXPORT_FMT`.
	- Plugins:
		- The I/O plugin interface has changed, so if you have your own I/O plugins, you will need to update them.
			- The interface name changed from `ccIOFilterPluginInterface` to `ccIOPluginInterface`.
			- The `ccIOPluginInterface::getFilter()` method was removed in favour of `ccIOPluginInterface::getFilters()`.
			- The `FileIOFilter` base class now takes a struct as an argument containing all the static info about a filter - extensions, features (import/export), etc.. See `FooFilter` in the `ExampleIOPlugin` and the comments in `FileIOFilter::FilterInfo`.
			- The use of `FileIOFilter::FilterInfo` means that the following virtual functions in I/O filters are no longer virtual/required:
				- importSupported
				- exportSupported
				- getFileFilters
				- getDefaultExtension
				- canLoadExtension
		- The GL plugin interface has changed, so if you have your own GL plugins, you will need to update them.
			- The interface name changed from `ccGLFilterPluginInterface` to `ccGLPluginInterface`.
	- CC will now handle external matrices (loaded or input via the 'Edit > Apply Transformation' tool) with a 16th component different than 0
		(this 16th component will be considered as the inverse scale)
	- Attempt to improve gamepads detection/connection

- Bug fixes:
	- repair ccPlane Entity bounding box showing bugs and normal vector showing bugs
	- LAS classification flags were not always properly extracted/saved by the standard LAS filter (depending on the point format)
	- Trace Polyline tool: when changing the OpenGL camera position while tracing a polyline AND using oversampling, strange spikes could appear
	- The Unroll dialog was not enabling all the apex coordinate fields after switching from Cylinder to Cone mode
	- The Clipping-box tool 'edit' dialog would sometimes move the box in an unexpected way when opening and closing it without making any change
	- M3C2: the 'subsampling' option was not properly restored when loading the parameters from a file (if 'SubsampleEnabled = false')
	- Orienting normals with a sensor position could lead to a crash
	- Shapefile: at export time, the SHX file (created next to the SHP file) was malformed
		- this prevented loading the file in most GIS tools
		- polylines were mistakenly exported as polygons
	- SRS (Spatial Reference System) information could be lost when loading LAS files
	- The cartesian bounding-box of exported E57 files was wrongly expressed in the file-level coordinate system (instead of the local one)
	- Data could be lost when merging two clouds with FWF data
	- When VBOs were activated with an ATI card, ACloudViewer could crash (because ATI only supports 32bit aligned VBOs :p)
	- The LAS 1.3/1.4 filter was not compressing files with a minor case 'laz' extension :(
	- The iteration stop criteria has been changed in the CSF plugin to fix a small bug

v3.4.0 (Asher) - (in development)
----------------------

- Enhancements
  - Speed up the Fit Sphere tool and point picking in the Registration Align (point pairs picking) tool
  - Translation:
	- new (argentinian) Spanish translation

- Bug fixes
  - Command line:
    - the 'EXTRACT_VERTICES' option was actually deleting the extracted vertices right after extracting them, causing a crash when trying to access them later :| (#847)
    - fix handling of SF indices in SF_ARITHMETIC and COMMAND_SF_OP
    - the COMMAND_ICP_ROT option of the ICP command line tool was ignored (#884)
    - when loading a BIN file from the command line, only the first-level clouds were considered
  - Fix loading LAS files with paths containing multi-byte characters when using PDAL (#869)
  - When saving a cloud read from LAS 1.0 let PDAL choose the default LAS version (#874)
  - Fix potential crash or use of incorrect data when comparing clouds (#871)
  - Fix potential crash when quitting or switching displays
  - Quitting the "Section extraction tool" (and probably any tool that uses a temporary 3D view, such as the Align tool) would break the picking hub mechanism (preventing the user from picking points typically) (#886)
  - Fix the camera name being displayed in the wrong place (#902)
  - The layers management of the Rasterize tool was partially broken
  - the C2C/C2M distance computation tool called through the command line was always displaying progress dialogs even in SILENT mode
  - the ICP registration tool called through the command line was always displaying progress dialogs even in SILENT mode
  - Fix potential crash with qCSF (see github issue #909)
  - In some cases, the (subsampled) core points cloud was not exported and saved at the end of the call to M3C2 through the command line
  - Some points were incorrectly removed by the 'Clean > Noise filer' method (parallelism issue)
  - The radius was not updated during the refinement pass of the Sphere fitting algorithm  (i.e. the final radius was not optimal)

v3.3.0 (Asher) - 24/02/2019
----------------------

- Bug fixes
  - Rasterize tool:
    - interpolating empty cells with the 'resample input cloud' option enabled would make CC crash
    - change layout so it works better on lower-resolution monitors
  - Command line:
    - the 'EXTRACT_VERTICES' option was not accessible
    - calling the -RASTERIZE option would cause an infinite loop
    - the Global Shift & Scale information of the input cloud was not transferred to the output cloud of the -RASTERIZE tool
  - glitch fix: the 3D window was not properly updated after rendering the screen as a file with a zoom > 1
  - glitch fix: the name of the entity was not displayed at the right place when rendering the screen as a file with a zoom > 1
  - the Surface and Volume Density features were potentially outputting incorrect values (the wrong source scalar field was used when applying the dimensional scale!)
  - the chosen octree level could be sub-optimal in some very particular cases
  - E57 pinhole images:
    - fix sensor array information (it was displaying total image size for the width of the image)
    - fix pixel width & height


- Translations
  - updated Russian translation (thanks to Eugene Kalabin)
  - added Japanese translation (thanks to the translators at CCCP)


- macOS Note
  - I (Andy) had to update ffmpeg, which is used by the animation plugin, for this patch release. Normally I would wait for 2.11, but homebrew changed their policies and started including everything in their build, so I can no longer use it. The good news is that compiling ffmpeg myself and statically linking shaves about 30 MB off the size of ACloudViewer.app...
  - it has been reported that this fixes a potential crash in ffmpeg's libavutil.56.dylib

v3.2.0 (Asher) - 01/16/2019
----------------------

- Bug fixes:

  - writing E57 files was broken
  - an exception was being thrown when you close CC after saving an ASCII file (#834)

v3.1.0 (Asher) - 01/06/2019
----------------------

- new features:

	* Edit > Polyline > Sample points
		- to regularly samples points on one or several polylines

	* New set of geometrical features to compute on clouds:
		- Tools > Other > Compute geometric features
		- features are all based on locally computed eigen values:
			* sum of eigen values
			* omnivariance
			* eigenentropy
			* anisotropy
			* planarity
			* linearity
			* PCA1
			* PCA2
			* surface variation
			* sphericity
			* verticality
		- most of the features are defined in "Contour detection in unstructured 3D point clouds", Hackel et al, 2016

	* Localization support
		- Display > Language translation
		- currently supported languages:
			* English (default)
			* Brazilian portuguese (partial)
			* French (very partial)
			* Russian (partial)
		- volunteers are welcome: https://www.CLOUDVIEWER .org/forum/viewtopic.php?t=1444

- enhancements:

	* Roughness, Density and Curvature can now all be computed via the new 'Tools > Other > Compute geometric features' menu
		(Approx density can't be computed anymore)

	* Global Shift & Scale dialog
		- new option "Preserve global shift on save" to directly control whether the global coordinates should be preserved
			at export time or simply forgotten

	* The 'Display > Lock vertical rotation' option has been renamed 'Display > Lock rotation about an axis' (Shortcut: L)
		- CC will now ask for the rotation axis to be locked (default: Z)

	* The M3C2 plugin can now be called from the command line:
		- the first time you'll need the configuration file saved with the GUI tool
			(Use the 'Save parameters to file' button in the bottom-left corner of the M3C2 dialog --> the floppy icon)
		- then load 2 files (cloud 1 and cloud2)
		- optionally load a 3rd cloud that will be used as core points
		- and eventually call the -M3C2 option with the parameter file as argument:
			ACloudViewer -O cloud1 -O cloud2 (-O core_points) -M3C2 parameters_file
		- new option to use the core points cloud normals (if any)

	* The Canupo plugin is now open-source!
		- Thanks (once again) to Dimitri Lague for this great contribution
		- the code is here: https://github.com/ACloudViewer/ACloudViewer/tree/master/plugins/core/qCanupo

	* The "Classify" option of the Canupo plugin can now be called from the command line:
		- you'll need a trained classifier (.prm file)
		- main option: -CANUPO_CLASSIFY classifier.prm
		- confidence threshold:
			* -USE_CONFIDENCE {threshold}  (threshold must be between 0 and 1)
			* (use the 'SET_ACTIVE_SF' after loading a cloud to set the active scalar field if
				you want it to be used to refine the classification)
		- syntax:
			ACloudViewer -O cloud1 ... -O cloudN -CANUPO_CLASSIFY (-USE_CONFIDENCE 0.9) classifier.prm

	* Labels can now be imported from ASCII files:
		- new column role in the ASCII loading dialog: "Labels"
		- labels can be created from textual or numerical columns
		- one "2D label" entity is created per point (don't try to load too many of them ;)
		- labels are displayed in 3D by default (i.e. next to each point), but they can also be displayed in 2D (see the dedicated check-box)

	* FBX units:
		- default FBX units are 'cm'
		- if a FBX file with other units is imported, CC will now store this information as meta-data and will set it correctly
			if the corresponding meshes are exported as FBX again

	* Command line mode:
		- scalar field convert to RGB:
			* '-SF_CONVERT_TO_RGB {mixWithExistingColors bool}'
		- scalar field set color scale:
			* '-SF_COLOR_SCALE {filename}'
		- extract all loaded mesh vertices as standalone 'clouds' (the mesh is discarded)
			* '-EXTRACT_VERTICES'
		- remove all scan grids
			* '-REMOVE_SCAN_GRIDS'
		- new sub-option of 'SAVE_CLOUDS' to set the output filename(s) (e.g. -SAVE_CLOUDS FILE "cloud1.bin cloud2.bin ..."
		- new options for the 'OCTREE_NORMALS' (thanks to Michael Barnes):
			* '-ORIENT' to specify a default orientation hint:
				- PLUS_ZERO
				- MINUS_ZERO
				- PLUS_BARYCENTER
				- MINUS_BARYCENTER
				- PLUS_X
				- MINUS_X
				- PLUS_Y
				- MINUS_Y
				- PLUS_Z
				- MINUS_Z
				- PREVIOUS
			* '-MODEL' to specify the local model:
				- LS
				- TRI
				- QUADRIC

	* Unroll tool:
		- the cylindrical unrolling can be performed inside an arbitrary angular range (between -3600 and +3600 degrees)
		- this means that the shape can be unrolled on more than 360 degrees, and from an arbitrary starting orientation

	* New options (Display > Display options):
		- the user can now control whether normals should be enabled on loaded clouds by default or not (default state is now 'off')
		- the user can now control whether load and save dialogs should be native ones or generic Qt dialogs

	* Normals:
		- ergonomics of 'Normals > compute' dialog have been (hopefully) enhanced
		- normals can now be oriented toward a sensor even if there's no grid associated to the point cloud.
		- the Normal Orientation algorithm based on the Minimum Spanning Tree now uses much less memory (~1/10)

	* PCV:
		- the PCV plugin can now be applied on several clouds (batch mode)

	* LAS I/O:
		- ACloudViewer can now read and save extra dimensions (for any file version) - see https://github.com/ACloudViewer/ACloudViewer/pull/666

	* E57:
		- the E57 plugin now uses [libE57Format] (https://github.com/asmaloney/libE57Format) which is a fork of the old E57RefImpl
		- if you compile ACloudViewer with the E57 plugin, you will need to use this new lib and change some CMake options to point at it - specifically **OPTION_USE_LIBE57FORMAT** and **LIBE57FORMAT_INSTALL_DIR**
		- the E57 plugin is now available on macOS

	* RDS (Riegl)
		- the reflectance scalar field read from RDS file should now have correct values (in dB)

	* SHP:
		- improved support thanks to T. Montaigu (saving and loading Multipatch entities, code refactoring, unit tests, etc.)

	* Cross section tool:
		- can now be started with a group of entities (no need to select the entities inside anymore)
		- produces less warnings

	* Plugins (General):
		- the "About Plugins" dialog was rewritten to provide more information about installed plugins and to include I/O and GL plugins.
		- [macOS] the "About Plugins..." menu item was moved from the Help menu to the Application menu.
		- added several fields to the plugin interface: authors, maintainers, and reference links.
		- I/O plugins now have the option to return a list of filters using a new method *getFilters()* (so one plugin can handle multiple file extensions)
		- moved support for several less frequently used file formats to a new plugin called qAdditionalIO
			- Snavely's Bundler output (*.out)
			- Clouds + calibrated images [meta][ascii] (*.icm)
			- Point + Normal cloud (*.pn)
			- Clouds + sensor info. [meta][ascii] (*.pov)
			- Point + Value cloud (*.pv)
			- Salome Hydro polylines (*.poly)
			- SinusX curve (*.sx)
			- Mensi Soisic cloud (*.soi)

	* Misc:
		- some loading dialogs 'Apply all' button will only apply to the set of selected files (ASCII, PLY and LAS)
		- the trace polyline tool will now use the Global Shift & Scale information of the first clicked entity
		- when calling the 'Edit > Edit Shift & Scale' dialog, the precision of the fields of the shift vector is now 6 digits
			(so as to let the user manually "geo-reference" a cloud)
		- the ASCII loading dialog can now load up to 512 columns (i.e. almost as many scalar fields ;). And it shouldn't become huge if
			there are too many columns or characters in the header line!

- bug fixes:

	* subsampling with a radius dependent on the active scalar field could make CC stall when dealing with negative values
	* point picking was performed on each click, even when double-clicking. This could actually prevent the double-click from
		being recognized as such (as the picking could be too slow!)
	* command line mode: when loading at least two LAS files with the 'GLOBAL_SHIFT AUTO' option, if the LAS files had different AND small LAS Shift
	* point picking on a mesh (i.e. mainly in the point-pair based registration tool) could select the wrong point on the triangle, or even a wrong triangle
	* raster I/O: when importing a raster file, the corresponding point cloud was shifted of half a pixel
	* the RASTERIZE command line could make CC crash at the end of the process
	* hitting the 'Apply all' button of the ASCII open dialog would not restore the previous load configuration correctly in all cases
		(the header line may not be extracted the second time, etc.)
	* align tool: large coordinates of manually input points were rounded off (only when displayed)
	* when applying an orthographic viewport while the 'stereo' mode is enabled, the stereo mode was broken (now a warning message is displayed and
		the stereo mode is automatically disabled)
	* the global shift along vertical dimension (e.g. Z) was not applied when exporting a raster grid to a raster file (geotiff)
	* the 2.5D Volume calculation tool was ignoring the strategy for filling the empty cells of the 'ceil' cloud (it was always using the 'ground' setting)
	* [macOS] fixed the squished text in the Matrix and Axis/Angle sections of the transformation history section of the properties
	* [macOS] fixed squished menus in the properties editor
	* the application options (i.e. only whether the normals should be displayed or not at loading time) were not saved!
	* DXF files generated by the qSRA plugin were broken (same bug as the DXF filter in version 2.9)
	* the OCTREE_NORMALS command was saving a file whatever the state of the AUTO_SAVE option
	* the Align tools could make CC crash when applying the alignment matrix (if the octree below the aligned entity was visible in the DB tree)
	* the clouds and contour lines generated by the Rasterize tool were shifted of half a cell
	* in some cases, merging a mesh with materials with a mesh without could make CC crash
	* command line mode: the VOLUME command parser would loop indefinitely if other commands were appended after its own options + it was ignoring the AUTO_SAVE state.
	* some files saved with version 2.6 to 2.9 and containing quadric primitives or projective camera sensors could not be loaded properly since the version 2.10.alpha of May 2018
	* for a mysterious reason, the FWF_SAVE_CLOUDS command was not accessible anymore...
	* when computing C2C distances, and using both a 2.5D Triangulation local model and the 'split distances along X, Y and Z' option, the split distances could be wrong in some times

v2.0.0 (Asher) - 11/03/2019
----------------------

- enhancements:

	* Primitive factory
		- sphere center can now be set before its creation (either manually, or via the clipboard if the string is 'x y z')

- Bug fixes:

	* DXF export was broken (styles table was not properly declared)
	* PLY files with texture indexes were not correctly read

v1.0.0 (Asher) - 10/22/2019
----------------------

- New features:

	* New plugin: qCompass
		- structural geology toolbox for the interpretation and analysis of virtual outcrop models (by Sam Thiele)
		- see http://www.CLOUDVIEWER .org/doc/wiki/index.php?title=Compass_(plugin)

	* 3D view pivot management:
		- new option to position the pivot point automatically on the point currently at the screen center (dynamic update)
			(now the default behavior, can be toggled thanks to the dedicated icon in the 'Viewing tools' toolbar or the 'Shift + P' shortcut)
		- double clicking on the 3D view will also reposition the pivot point on the point under the cursor
		- the state of this option is automatically saved and restored when CC starts

	* New tool to import scalar fields from one cloud to another: 'Edit > SFs > Interpolate from another entity'
		- 3 neighbor extraction methods are supported (nearest neighbor, inside a sphere or with a given number of neighbors)
		- 3 algorithms are available: average, median and weighted average

	* New sub-menu 'Tools > Batch export'
		- 'Export cloud info' (formerly in the 'Sand-box' sub-menu)
			* exports various pieces of information about selected clouds in a CSV file
			* Name, point count, barycenter
			+ for each scalar field: name, mean value, std. dev. and sum
		- 'Export plane info'
			* exports various pieces of information about selected planes in a CSV file
			* Name, width, height, center, normal, dip and dip direction

	* New interactor to change the default line width (via the 'hot zone' in the upper-left corner of 3D views)

	* New option: 'Display > Show cursor coordinates'
		- if activated, the position of the mouse cursor relatively to the 3D view is constantly displayed
		- the 2D position (in pixels) is always displayed
		- the 3D position of the point below the cursor is displayed if possible

	* New shortcut: P (pick rotation center)

- enhancements:

	* When a picking operation is active, the ESC key will cancel it.

	* qBroom plugin:
		- now has a wiki documentation: http://asher-1.github.io/docs

	* qAnimation plugin:
		- new output option 'zoom' (alternative to the existing 'super resolution' option)
		- the plugin doesn't spam the Console at each frame if the 'super resolution' option is > 1 ;)

	* M3C2 plugin:
		- "Precision Maps" support added (as described in "3D uncertainty-based topographic change detection with SfM
			photogrammetry: precision maps for ground control and directly georeferenced surveys" by James et al.)
		- Allows for the computation of the uncertainty based on precision scalar fields (standard deviation along X, Y and Z)
			instead of the cloud local roughness

	* 'Unroll' tool:
		- new cone 'unroll' mode (the true 'unroll' mode - the other one has been renamed 'Straightened cone' ;)
		- option to export the deviation scalar-field (deviation to the theoretical cylinder / cone)
		- dialog parameters are now saved in persistent settings

	* Plugins can now be called in command line mode
		(the 'ccPluginInterface::registerCommands' method must be reimplemented)
		(someone still needs to do the job for each plugin ;)

	* Trace polyline tool
		- the tool now works on meshes
		- Holding CTRL while pressing the right mouse button will pan the view instead of closing the polyline
		- new 'Continue' button, in case the user has mistakenly closed the polyline and wants to continue

	* Command line mode
		- the Rasterize tool is now accessible via the command line:
			* '-RASTERIZE -GRID_STEP {value}'
			* additional options are:
				-VERT_DIR {0=X/1=Y/2=Z} - default is Z
				-EMPTY_FILL {MIN_H/MAX_H/CUSTOM_H/INTERP} - default is 'leave cells empty'
				-CUSTOM_HEIGHT {value} - to define the custom height filling value if the 'CUSTOM_H' strategy is used (see above)
				-PROJ {MIN/AVG/MAX} - default is AVG (average)
				-SF_PROJ {MIN/AVG/MAX} - default is AVG (average)
				-OUTPUT_CLOUD - to output the result as a cloud (default if no other output format is defined)
				-OUTPUT_MESH - to output the result as a mesh
				-OUTPUT_RASTER_Z - to output the result as a geotiff raster (altitudes + all SFs by default, no RGB)
				-OUTPUT_RASTER_RGB - to output the result as a geotiff raster (RGB)
				-RESAMPLE - to resample the input cloud instead of generating a regular cloud (or mesh)
			* if OUTPUT_CLOUD and/or OUTPUT_MESH options are selected, the resulting entities are kept in memory.
				Moreover if OUTPUT_CLOUD is selected, the resulting raster will replace the original cloud.
		- 2.5D Volume Calculation tool
			* '-VOLUME -GRID_STEP {...} etc.' (see the wiki for more details)
		- Export coord. to SF
			* '-COORD_TO_SF {X, Y or Z}'
		- Compute unstructured cloud normals:
			* '-OCTREE_NORMALS {radius}'
			* for now the local model is 'Height Function' and no default orientation is specified
		- Clear normals
			* '-CLEAR_NORMALS'
		- New mesh merging option
			* '-MERGE_MESHES'
		- Compute mesh volume:
			* '-MESH_VOLUME'
			* optional argument: '-TO_FILE {filename}' to output the volume(s) in a file
		- LAS files:
			* when loading LAS files without any specification about Global Shift, no shift will be applied, not even the LAS file internal 'shift' (to avoid confusion)
			* however, it is highly recommanded to always specifiy a Global Shift (AUTO or a specific vector) to avoid losing precision when dealing with big coordinates!
		- Other improvements:
			* the progress bar shouldn't appear anymore when loading / saving a file with 'SILENT' mode enabled
			* the ASCII loading dialog shouldn't appear anymore in 'SILENT' mode (only if CC really can't guess anything)
			* the default timestamp resolution has been increased (with milliseconds) in order to avoid overwriting files
				when saving very small file (too quickly!)

	* Rasterize tool
		- contour lines generation is now based on GDAL (more robust, proper handling of empty cells, etc.)
		- new option to re-project contour lines computed on a scalar field (i.e. a layer other than the altitudes)
			on the altitudes layer
		- the grid step bounds have been widened (between 1E-5 and 1E+5)

	* Edit > SF > Compute Stat. params
		- the RMS of the active SF is now automatically computed and displayed in the Console

	* PLY I/O filter
		- now supports quads (quads are loaded as 2 triangles)

	* DXF I/O filter
		- now based on dxflib 3.17.0
		- point clouds can now be exported to DXF (the number of points should remain very limited)
		- see fixed bugs below

	* LAS I/O filter
		- the 'Spatial Reference System' of LAS files is now stored as meta-data and restored
			when exporting the cloud as a LAS/LAZ file.

	* [Windows] qLAS_FWF:
		- the plugin (based on LASlib) can now load most of the standard LAS fields
		- the plugin can now save files (with or without waveforms)
		- the plugin can now be called in command line mode:
			-FWF_O: open a LAS 1.3+ file
			-FWF_SAVE_CLOUDS: save cloud(s) to LAS 1.3+ file(s) (options are 'ALL_AT_ONCE' and 'COMPRESSED' to save LAZ files instead of LAS)

	* New method: 'Edit > Waveforms > Compress FWF data'
		- To compress FWF data associated to a cloud (useful after a manual segmentation for instance
			as the FWF data is shared between clouds and remains complete by default)
		- Compression is done automatically when saving a cloud with the 'LAS 1.3 / 1.4' filter (QLAS_FWF_IO_PLUGIN)
			(but it's not done when saving the entity as a BIN file)

	* Oculus support
		- CC now displays in the current 3D view the mirror image of what is displayed in the headset
		- using SDK 1.15

	* Point List Picking tool
		- the list can now be exported as a 'global index, x, y, z' text file

	* Scale / Multiply dialog:
		- new option to use the same scale for all dimensions
		- new option to apply the scale to the 'Global shift' (or not)

	* New Menu Entry: 'Edit > Grid > Delete scan grids'
		- scan grids associated to a cloud can now be deleted (to save space when saving the cloud to a BIN file for instance)

	* qEllipser plugin:
		- option to export the image as a (potentially scaled) point cloud

	* Normal computation tool:
		- new algorithm to compute the normals based on scan grids (faster, and more robust)
		- the 'kernel size' parameter is replaced by 'the minimum angle of triangles' used in the internal triangulation process
		- Plane and Quadric modes will now automatically increase the radius adaptively to reach a minimum number of points and to avoid creating 'zero' (invalid) normals

	* Edit the scalar value of a single point
		- create a label on the point (SHIFT + click)
		- make sure a scalar field is active
		- right click on the label entry in the DB tree and select 'Edit scalar value'

	* Merge (clouds)
		- new option to generate a scalar field with the index of the original cloud for each point

	* Other
		- color scales are now listed in alphabetical order
		- polylines exported from the 'Interactive Segmentation' tool will now use the same Global Shift as the segmented entity(ies)
		- when changing the dip and dip direction of plane parallel with XY, the resulting plane shouldn't rotate in an arbitrary way anymore
		- the filter and single-button plugin toolbars are now on the right side of the window by default (to reset to the default layouts, use "reset all GUI element positions" at the bottom of the Display menu)
		- the Plane edition dialog now lest the user specify the normal plane in addition to its dip and dip direction
		- new 'Clone' icon with a colored background so as to more clearly spot when the icon is enabled (Nyan sheep!)
		- now using PoissonRecon 9.011
		- the default maximum point size and maximum line width increased to 16 pixels

- Bug fixes:
	* STL files are now output by default in BINARY mode in command line mode (no more annoying dialog)
	* when computing distances, the octree could be modified but the LOD structure was not updated
		(resulting in potentially heavy display artifacts)
	* glitch fix: the 'SF > Gradient' tool was mistakenly renaming the input scalar field ('.gradient' appended)
	* glitch fix: the picking process was ignoring the fact that meshes could be displayed in wireframe mode (they are now ignored in this case)
	* command line 'CROSS_SECTION' option: the repetition of the cuts (<RepeatDim> option) could be incomplete in some cases (some chunks were missing)
	* raster loading: rasters loaded as clouds were shifted of half a pixel
	* the 'Edit > Sensors > Camera > Create' function was broken (input parameters were ignored)
	* merging clouds with FWF data would duplicate the waveforms of the first one
	* invalid lines in ASCII (text) files could be considered as a valid point with coordinates (0, 0, 0)
	* Point-pair based alignment tool:
		- extracting spheres on a cloud with Global Shift would create the sphere in the global coordinate system instead of the local one (i.e. the sphere was not visible)
		- deleting a point would remove all the detected spheres
	* The FARO I/O plugin was associating a wrong transformation to the scan grids, resulting in weird results when computing normals or constructing a mesh based on scan grids
	* When editing only the dip / dip direction of a plane, the rotation was not made about the plane center
	* qSRA plugin: profile polyline automatically generated from cylinders or cone were shifted (half of the cylinder/cone height), resulting in a 'shifted' distance map
		(half of the cloud was 'ignored')
	* DXF export
		- the I/O filter was mistakenly exporting the vertices of polylines and meshes as separate clouds
		- the I/O filter was not exporting the shifted point clouds at the right location
	* Render to file:
		- when the 'draw rounded points' option was enabled, pixel transparency could cause a strange effect when exported to PNG images
	* Octree rendering:
		- the 'Cube' mode was not functional
		- the 'Point' mode with normals was not functional