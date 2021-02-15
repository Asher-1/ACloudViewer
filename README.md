
![](./doc/ErowCloudViewer_logo_horizontal.png)

# ErowCloudViewer: A Modern System for 3D Data Processing

<h4>
    <a href="http://www.erow.cn">Homepage</a> |
    <a href="http://www.erow.cn/docs">Docs</a> |
    <a href="http://www.erow.cn/docs/release/getting_started.html">Quick Start</a> |
    <a href="http://www.erow.cn/docs/release/compilation.html">Compile</a> |
    <a href="http://www.erow.cn/docs/release/index.html#python-api-index">Python</a> |
    <a href="http://www.erow.cn/docs/release/cpp_api.html">C++</a> |
    <a href="https://github.com/Asher-1/CloudViewer-ML">CloudViewer-ML</a> |
    <a href="https://github.com/Asher-1/ErowCloudViewer/releases">Viewer</a> |
    <a href="http://www.erow.cn/docs/release/contribute/contribute.html">Contribute</a> |
    <a href="https://www.youtube.com/channel/UCRJBlASPfPBtPXJSPffJV-w">Demo</a> |
    <a href="https://forum.erow.cn">Forum</a>
</h4>

[![GitHub release](./doc/version.svg)](https://github.com/Asher-1/ErowCloudViewer/releases/)
[![Build Status](./doc/ErowCloudViewer.svg)](https://github.com/Asher-1/ErowCloudViewer/releases/)
[![Releases](./doc/newRelease.svg)](https://github.com/Asher-1/ErowCloudViewer/releases/)

Introduction
------------
ErowCloudViewer is an open-source library that supports rapid development of software
that deals with 3D data. The ErowCloudViewer frontend exposes a set of carefully selected
data structures and algorithms in both C++ and Python. The backend is highly optimized and is set up for parallelization. We welcome contributions from the open-source community.

------------
ErowCloudViewer is a 3D point cloud (and triangular mesh) processing software.
It was originally designed to perform comparison between two 3D points clouds
(such as the ones obtained with a laser scanner) or between a point cloud and a
triangular mesh. It relies on an octree structure that is highly optimized for
this particular use-case. It was also meant to deal with huge point
clouds (typically more than 10 millions points, and up to 120 millions with 2 Gb
of memory).

More on ErowCloudViewer [here](http://www.erow.cn)

**Core features of ErowCloudViewer include:**

* 3D data structures
* 3D data processing algorithms
* Scene reconstruction
* Surface alignment
* 3D visualization
* Physically based rendering (PBR)
* Available in C++ and Python

For more, please visit the [ErowCloudViewer documentation](http://www.erow.cn).

Compilation
-----------

Supported OS: Windows, Linux, and Mac OS X

Refer to the [compiling-cloudviewer.md file](doc/compiling-cloudviewer.md) for up-to-date information.

Refer to the [BUILD.md file](BUILD.md) for up-to-date information.

Basically, you have to:
- clone this repository
- install mandatory dependencies (OpenGL,  etc.) and optional ones if you really need them
(mainly to support particular file formats, or for some plugins)
- launch CMake (from the trunk root)
- enjoy!


Contributing to ErowCloudViewer
----------------------------

If you want to help us improve ErowCloudViewer or create a new plugin you can start by reading this [guide](CONTRIBUTING.md)

Supporting the project
----------------------

If you want to help us in another way, you can make donations via [donorbox](https://www.erow.cn)
Thanks!

[![donorbox](./doc/button-medium-blue.png)](https://www.erow.cn)
