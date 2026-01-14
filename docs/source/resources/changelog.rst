Changelog
=========

For the complete changelog, please see the `CHANGELOG.md <https://github.com/Asher-1/ACloudViewer/blob/main/CHANGELOG.md>`_ file in the repository.

Recent Releases
---------------

v3.9.4-Beta (Latest Release - Feb 12, 2025)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**New Features:**

- VTK PBR rendering support for multiple texturing
- Texturing Atlas support for reconstruction module
- Excellent Texture mesh rendering support
- 2D Image Rendering with transparency feature
- Qt6 compatible interface strategy
- Cork plugin for Linux and macOS
- Ruler, protractor, and contour widgets (VTK-based measurement tools)
- Selection tools based on VTK
- Color filtering (Gaussian, Bilateral, Median, Mean filters)
- Custom keyboard shortcuts setting dialog
- New entities: Circle and Disc
- View property for ccObject
- Light specular intensity adjustment
- AxesGrid support for ccObject
- Documentation support

**Bug Fixes:**

- Fixed Illegal instruction issues with PCL and ACloudViewer
- Fixed crash on Ubuntu 22.04 for reconstruction due to misalignment
- Fixed Python env package issues with pyenv manager
- Fixed numpy version > 2.x issues
- Fixed ReconstructionOptionsWidget crash issues

**Enhancements:**

- Replace `#ifndef` with `#pragma once`
- Use GLIBCXX_USE_CXX11_ABI=ON by default
- CUDA support 11.8 → 12.6.3, TensorFlow 2.16.2 → 2.19.0, PyTorch 2.2.2 → 2.7.1
- Change python package file name from `cloudViewer*` to `cloudviewer*` (PEP 503)
- Code style check support

**New Plugins:**

- G3 Point - Granulometry made simple

**Supported Platforms:**

- Windows x86/64
- Linux x86/64
- macOS X64 && arm64 (M1-4)

v3.9.3 (Oct 14, 2025)
^^^^^^^^^^^^^^^^^^^^^

**New Features:**

- ScaleBar support
- Doppler ICP in tensor registration pipeline
- In-memory loading of XYZ files
- Python pathlib support for file IO
- Albedo texture to mesh from calibrated images
- FlyingEdges algorithm for isosurface extraction
- O3DVisualizer API improvements
- Radio button support
- Updated GLFW and jupyter web_visualizer

**Bug Fixes:**

- Fixed build with librealsense v2.44.0 and VS 2022 17.13
- Fixed Windows CI issues (VS 16 2019 → VS 17 2022)
- Fixed display issues with text on macOS
- Fixed TriangleMesh sampling methods
- Fixed tensor-based TSDF integration
- Fixed geometry picker errors with LineSet objects
- Fixed render to depth image on Apple Retina displays
- Fixed missing liblapack.so issues
- Fixed CloudViewerApp no GUI display issues

**Enhancements:**

- Updated 3rdparty versions
- Exposed more functionality in SLAM and odometry pipelines
- Refactored python examples directory
- Robust image I/O
- Default dataset loading in reconstruction system

**Supported Platforms:**

- Windows x86/64
- Linux x86/64
- macOS X64 && arm64 (M1-4)

v3.9.2 (Dec 22, 2024)
^^^^^^^^^^^^^^^^^^^^^

**New Features:**

- Save project menu entry (File > Save project or CTRL+SHIFT+S)
- Tools > Fit > circle (fits 2D circle on 3D point cloud)
- CI support via GitHub Actions (macOS, Windows, Linux)

**New Plugins:**

- qLASIO - Unified LAS file loader based on LASzip
- VoxFall - Volumetric change detection for rockfalls
- q3DMASC - 3D point cloud classification
- qTreeIso - 3D graph-based individual-tree isolator
- qPythonRuntime - Python automation support
- 3DFin - 3D Forest Inventory (Python-based)

**Enhancements:**

- Command line mode improvements (I/O plugins loaded)
- PCD format improvements (integer and double coordinates, scan grids)
- Animation plugin now uses ffmpeg 6.1

**Supported Platforms:**

- Windows x86/64
- Linux x86/64
- macOS X64 && arm64 (M1-2)

v3.9.1 (Nov 15, 2024)
^^^^^^^^^^^^^^^^^^^^^

**Highlights:**

- Linux auto CI support for Ubuntu 18.04-22.04
- Fully supported Docker building
- Updated PCL version from 1.11.1 to 1.14.1
- Updated VTK version from 8.2.0 to 9.3.1
- Migrated from C++14 to C++17 as default
- Python 3.12 support

**Supported Platforms:**

- Windows x86/64
- Linux x86/64
- macOS arm64 (M1-2)

v3.9.0 (Apr 5, 2023)
^^^^^^^^^^^^^^^^^^^^

**New Features:**

- Lock rotation axis feature
- Animation display features
- Scalar field switch feature
- PLY mesh texture_u and texture_v support
- New platform support: macOS

**Supported Platforms:**

- Windows x86/64
- Linux x86/64
- macOS arm64 (M1-2)

v3.8.0 (Oct 10, 2021)
^^^^^^^^^^^^^^^^^^^^^

**Major Features:**

- New 3D reconstruction system based on COLMAP
- Real-time 3D reconstruction pipeline (VoxelHashing with GPU/CPU)
- Real-time point cloud registration (high-performance ICP)
- New Neighbor Search module (KNN, RadiusSearch, GPU/CPU support)
- Web visualizer for Jupyter and web environments
- 3D machine learning models (PointRCNN, SparseConvNets)
- Upgraded GUI module with new widgets
- CUDA 11 support

**Supported Platforms:**

- Windows x86/64
- Linux x86/64
- Linux ARM64

Release Schedule
----------------

ACloudViewer follows semantic versioning (``MAJOR.MINOR.PATCH``):

- **Major**: Breaking API changes
- **Minor**: New features, backward compatible
- **Patch**: Bug fixes, backward compatible

Beta releases are tagged with ``Beta`` suffix.

Getting Release Notifications
------------------------------

To stay updated on new releases:

1. **Watch the Repository**: Click "Watch" on `GitHub <https://github.com/Asher-1/ACloudViewer>`_
2. **Release Notes**: Check `GitHub Releases <https://github.com/Asher-1/ACloudViewer/releases>`_
3. **RSS Feed**: Subscribe to the releases RSS feed

Version Support
---------------

- **Latest Stable**: Full support with bug fixes and security updates
- **Previous Minor**: Security updates only
- **Older Versions**: Community support, no official updates

Upgrading
---------

When upgrading between versions:

1. **Review the Changelog**: Check for breaking changes
2. **Update Dependencies**: Ensure compatible library versions
3. **Test Thoroughly**: Run your test suite after upgrading
4. **Report Issues**: File bugs if you encounter problems

See Also
--------

- :doc:`/getting_started/build_from_source` - Building from source
- :doc:`faq` - Frequently Asked Questions
- :doc:`support` - Getting help
