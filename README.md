<p align="center">
  <img src="docs/images/ACloudViewer_logo_horizontal.png">
</p>

# ACloudViewer: A Modern System for 3D Data Processing

<h4>
    <a href="https://asher-1.github.io/ACloudViewer/docs">Homepage</a> |
    <a href="https://asher-1.github.io/ACloudViewer/docs#about">About</a> |
    <a href="https://asher-1.github.io/ACloudViewer/docs#quickstart">Quick Start</a> |
    <a href="https://asher-1.github.io/ACloudViewer/docs#download">Download</a> |
    <a href="https://asher-1.github.io/ACloudViewer/docs/guides/building/">Build Guide</a> |
    <a href="https://asher-1.github.io/ACloudViewer/docs#gallery">Gallery</a> |
    <a href="https://github.com/Asher-1/CloudViewer-ML">CloudViewer-ML</a> |
    <a href="https://github.com/Asher-1/ACloudViewer/releases">Releases</a> |
    <a href="https://github.com/Asher-1/ACloudViewer/blob/main/CONTRIBUTING.md">Contribute</a> |
    <a href="https://www.youtube.com/channel/UCRJBlASPfPBtPXJSPffJV-w">Demo</a>
</h4>


[![GitHub release](docs/images/version.svg)](https://github.com/Asher-1/ACloudViewer/releases/)
[![Ubuntu CI](https://github.com/Asher-1/ACloudViewer/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/Asher-1/ACloudViewer/actions?query=workflow%3A%22Ubuntu+CI%22)
[![macOS CI](https://github.com/Asher-1/ACloudViewer/actions/workflows/macos.yml/badge.svg)](https://github.com/Asher-1/ACloudViewer/actions?query=workflow%3A%22macOS+CI%22)
[![Windows CI](https://github.com/Asher-1/ACloudViewer/actions/workflows/windows.yml/badge.svg)](https://github.com/Asher-1/ACloudViewer/actions?query=workflow%3A%22Windows+CI%22)
[![Releases](docs/images/newRelease.svg)](https://github.com/Asher-1/ACloudViewer/releases/)

Introduction
------------
ACloudViewer is an open-source library that supports rapid development of software that deals with 3D data 
which is highly based on CloudCompare, Open3D, Paraview and colmap with PCL. The ACloudViewer frontend exposes a set of 
carefully selected data structures and algorithms in both C++ and Python. The backend is highly optimized and 
is set up for parallelization. We welcome contributions from the open-source community.

------------
ACloudViewer is a 3D point cloud (and triangular mesh) processing software. It was originally designed to perform
comparison between two 3D points clouds
(such as the ones obtained with a laser scanner) or between a point cloud and a triangular mesh. It relies on an octree
structure that is highly optimized for this particular use-case. It was also meant to deal with huge point clouds (
typically more than 10 millions points, and up to 120 millions with 2 Gb of memory).

More on ACloudViewer [here](https://asher-1.github.io/ACloudViewer/docs)

**Core features of ACloudViewer include:**

* 3D data structures
* 3D data processing algorithms
* Scene reconstruction (based on colmap)
* Surface alignment
* 3D visualization
* Physically based rendering (PBR)
* 3D machine learning support with PyTorch and TensorFlow
* GPU acceleration for core 3D operations
* Available in C++ and Python

Here's a brief overview of the different components of ACloudViewer and how they fit
together to enable full end to end pipelines:

![CloudViewer_layers](docs/images/AbstractionLayers.png)

For more, please visit the [ACloudViewer documentation](https://asher-1.github.io/ACloudViewer/docs).


## Python quick start

Pre-built pip packages support Ubuntu 20.04+, macOS 10.15+ and Windows 10+
(64-bit) with Python 3.10-3.12 and cuda12.x.

```bash
# Install
pip install cloudViewer       # or
pip install cloudViewer-cpu   # Smaller CPU only wheel on x86_64 Linux (v3.9.1+)

# Verify installation
python -c "import cloudViewer as cv3d; print(cv3d.__version__)"

# Python API
python -c "import cloudViewer as cv3d; \
           mesh = cv3d.geometry.ccMesh.create_sphere(); \
           mesh.compute_vertex_normals(); \
           cv3d.visualization.draw(mesh, raw_mode=True)"

# CloudViewer CLI
cloudViewer example visualization/draw

# CloudViewer Reconstruction
cloudViewer example reconstruction/gui
```

## ACloudViewer System

<p align="center">
  <img width="640" src="docs/images/Annotaion.png">
</p>

ACloudViewer is a standalone 3D viewer app based on QT5 available on Ubuntu and Windows.
Please stay tuned for MacOS. Download ACloudViewer from the [release page](https://github.com/Asher-1/ACloudViewer/releases).

Semantic Annotation Tool:

<p align="center">
  <img width="640" src="docs/images/SemanticAnnotation.png" >
</p>


Reconstruction Tool:

<p align="center">
  <img width="640" src="docs/images/Reconstruction.png" >
</p>



## CloudViewer app

<p align="center">
  <img width="640" src="docs/images/CloudViewerApp.png">
</p>

CloudViewer-Viewer is a standalone 3D viewer app available on Ubuntu and Windows.
Please stay tuned for MacOS. Download CloudViewer app from the
[release page](https://github.com/Asher-1/ACloudViewer/releases).

## CloudViewer-ML

<p align="center">
  <img width="640" src="docs/images/gifs/getting_started_ml_visualizer.gif">
</p>

CloudViewer-ML is an extension of CloudViewer for 3D machine learning tasks. It builds on
top of the CloudViewer core library and extends it with machine learning tools for
3D data processing. To try it out, install CloudViewer with PyTorch or TensorFlow and check out
[CloudViewer-ML](https://github.com/Asher-1/CloudViewer-ML).

Compilation
-----------

Supported OS: Windows, Linux, and Mac OS X

Refer to the [BUILD.md file](BUILD.md) for detailed build instructions.

Online compilation guides:
- [Linux Build Guide](https://asher-1.github.io/ACloudViewer/docs/guides/building/compiling-cloudviewer-linux.html)
- [macOS Build Guide](https://asher-1.github.io/ACloudViewer/docs/guides/building/compiling-cloudviewer-macos.html)
- [Windows Build Guide](https://asher-1.github.io/ACloudViewer/docs/guides/building/compiling-cloudviewer-windows.html)

Basically, you have to:

- clone this repository
- install mandatory dependencies (OpenGL, etc.) and optional ones if you really need them
  (mainly to support particular file formats, or for some plugins)
- launch CMake (from the trunk root)
- enjoy!

Contributing to ACloudViewer
----------------------------

If you want to help us improve ACloudViewer or create a new plugin you can start by reading
this [guide](CONTRIBUTING.md)

Supporting the project
----------------------

If you want to help us in another way, you can make donations via [donorbox](https://asher-1.github.io/ACloudViewer/docs)
Thanks!

[![donorbox](docs/images/button-medium-blue.png)](https://asher-1.github.io/ACloudViewer/docs)
