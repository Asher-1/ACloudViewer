<p align="center">
<img src="https://github.com/Asher-1/ErowCloudViewer/blob/master/doc/ErowCloudViewer_logo_horizontal.png" width="512" />
</p>

# ErowCloudViewer: A Modern Library for 3D Data Processing

Homepage: http://www.erow.cn/

[![GitHub release](https://img.shields.io/github/release/cloudcompare/trunk.svg)](https://github.com/Asher-1/ErowCloudViewer/releases/)
[![Build Status](https://travis-ci.org/CloudCompare/CloudCompare.svg?branch=master)](https://travis-ci.org/CloudCompare/CloudCompare)
[![Releases](https://coderelease.io/badge/CloudCompare/CloudCompare)](https://coderelease.io/github/repository/CloudCompare/CloudCompare)

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

More on ErowCloudViewer [here](http://en.wikipedia.org/wiki/CloudCompare)

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

If you want to help us in another way, you can make donations via [donorbox](https://donorbox.org/support-cloudcompare)
Thanks!

<a href='https://donorbox.org/support-cloudcompare' target="_blank"><img src="https://d1iczxrky3cnb2.cloudfront.net/button-medium-blue.png"></a>

有道云翻译：
应用ID：6a470044a4c9069a
应用秘钥：62EddBoyfG0KXCp5Ih0C100Ir0DNYcId

翻译文件的正则替换：
ui_(\w+).h
\1.ui

"(\w+).cpp"
"../\1.cpp"
"(\w+).h"
"../\1.h"

Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Debug"
Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Debug\plugins"

Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Release"
Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Release\plugins"

Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Debug"
Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Debug\plugins"
Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Release"
Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Release\plugins"
