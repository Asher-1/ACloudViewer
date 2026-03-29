<p align="center">
  <img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/ACloudViewer_logo_horizontal.png">
</p>

# ACloudViewer: A Modern System for 3D Data Processing

<h4>
    <a href="https://asher-1.github.io/ACloudViewer/">Homepage</a> |
    <a href="https://asher-1.github.io/CLI-Anything/">CLI-Anything</a> |
    <a href="https://asher-1.github.io/ACloudViewer/#quickstart">Quick Start</a> |
    <a href="https://asher-1.github.io/ACloudViewer/#download">Download</a> |
    <a href="https://asher-1.github.io/ACloudViewer/documentation/getting_started/build_from_source.html">Build Guide</a> |
    <a href="https://asher-1.github.io/ACloudViewer/#gallery">Gallery</a> |
    <a href="https://github.com/Asher-1/CloudViewer-ML">CloudViewer-ML</a> |
    <a href="https://github.com/Asher-1/ACloudViewer/releases">Releases</a> |
</h4>


[![GitHub release](https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/version.svg)](https://github.com/Asher-1/ACloudViewer/releases/)
[![Ubuntu CI](https://github.com/Asher-1/ACloudViewer/actions/workflows/ubuntu.yml/badge.svg)](https://github.com/Asher-1/ACloudViewer/actions?query=workflow%3A%22Ubuntu+CI%22)
[![macOS CI](https://github.com/Asher-1/ACloudViewer/actions/workflows/macos.yml/badge.svg)](https://github.com/Asher-1/ACloudViewer/actions?query=workflow%3A%22macOS+CI%22)
[![Windows CI](https://github.com/Asher-1/ACloudViewer/actions/workflows/windows.yml/badge.svg)](https://github.com/Asher-1/ACloudViewer/actions?query=workflow%3A%22Windows+CI%22)
[![Releases](https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/newRelease.svg)](https://github.com/Asher-1/ACloudViewer/releases/)

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

More on ACloudViewer [here](https://asher-1.github.io/ACloudViewer/)

**Core features of ACloudViewer include:**

* 3D data structures
* 3D data processing algorithms
* Scene reconstruction (based on colmap)
* **3D Gaussian Splatting** real-time rendering and novel view synthesis (SIBR plugin)
* Surface alignment
* 3D visualization
* Physically based rendering (PBR)
* 3D machine learning support with PyTorch and TensorFlow
* GPU acceleration for core 3D operations
* Available in C++ and Python

Here's a brief overview of the different components of ACloudViewer and how they fit
together to enable full end to end pipelines:

![CloudViewer_layers](https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/AbstractionLayers.png)

For more, please visit the [ACloudViewer documentation](https://asher-1.github.io/ACloudViewer/documentation/).


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
  <img width="640" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/MainUI.png">
</p>

ACloudViewer is a standalone 3D viewer app based on QT5 available on Ubuntu and Windows.
Please stay tuned for MacOS. Download ACloudViewer from the [release page](https://github.com/Asher-1/ACloudViewer/releases).

---

#### Semantic Annotation

<table>
<tr>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/Annotaion.png" width="100%">
<br><sub><b>Annotation Interface</b></sub>
</td>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/SemanticAnnotation.png" width="100%">
<br><sub><b>Semantic Labeling</b></sub>
</td>
</tr>
</table>

---

#### Reconstruction & Measurement

<table>
<tr>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/Reconstruction.png" width="100%">
<br><sub><b>Reconstruction Tool</b></sub>
</td>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/selection_tools.png" width="100%">
<br><sub><b>Selection Tools</b></sub>
</td>
</tr>
<tr>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/ruler_measurement_tools.png" width="100%">
<br><sub><b>Ruler Measurement</b></sub>
</td>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/protractor_measurement_tools.png" width="100%">
<br><sub><b>Protractor Measurement</b></sub>
</td>
</tr>
</table>


## SIBR Viewer — 3D Gaussian Splatting & Novel View Synthesis

<p align="center">
  <img width="640" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/SIBR_viewer.png">
</p>

ACloudViewer integrates the **SIBR framework** (System for Image-Based Rendering) as a built-in plugin,
enabling real-time **3D Gaussian Splatting** rendering and novel view synthesis directly within the application.

Key capabilities:
* **3D Gaussian Splatting Viewer** — Real-time CUDA-accelerated rendering of trained 3DGS models
* **Remote Gaussian Viewer** — Live connection to a running 3DGS training process for monitoring
* **ULR / ULR v2 Viewer** — Unstructured Lumigraph Rendering for novel view synthesis
* **Textured Mesh & Point-Based Viewers** — View IBR datasets with scene debug overlays
* **Bidirectional interaction** — Select entities in ACloudViewer, auto-detect best viewer, import results back with auto-zoom

Enable with `-DPLUGIN_STANDARD_QSIBR=ON -DBUILD_CUDA_MODULE=ON` (CUDA optional for non-3DGS viewers).
See [qSIBR plugin documentation](plugins/core/Standard/qSIBR/README.md) for details.

## AI Agent Integration

ACloudViewer can be controlled by AI agents (OpenClaw, Cursor, Claude Code) through
a unified integration layer providing three interfaces:

| Interface | Protocol | Use Case |
|-----------|----------|----------|
| **JSON-RPC Plugin** | WebSocket (port 6001) | Real-time GUI control from agents |
| **MCP Server** | Model Context Protocol (stdio) | OpenClaw, Cursor IDE, Claude Code |
| **CLI Harness** | Click CLI + REPL | Shell scripts, headless batch processing |

```bash
# Install the CLI harness (headless or GUI control)
pip install git+https://github.com/Asher-1/CLI-Anything.git#subdirectory=acloudviewer/agent-harness

# Convert point cloud formats
cli-anything-acloudviewer --mode headless convert input.ply output.pcd

# Subsample with voxel grid
cli-anything-acloudviewer --mode headless process subsample input.ply -o sub.ply --voxel-size 0.05

# Start MCP server for AI agent frameworks
cli-anything-acloudviewer-mcp --mode auto
```

The headless mode invokes the ACloudViewer binary directly (no Python bindings needed),
supporting all 40+ file formats including plugin-provided ones (LAS/LAZ, E57, FBX, PCD, Draco).

Enable the JSON-RPC plugin with `-DPLUGIN_STANDARD_QJSONRPC=ON` at build time.

Browse available tools on the [CLI-Anything Hub](https://asher-1.github.io/CLI-Anything/).
See [agent-integration/](agent-integration/) for full documentation, MCP tool reference,
and the unified test suite.

## CloudViewer App & CloudViewer-ML

<table>
<tr>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/CloudViewerApp.png" width="100%">
<br><sub><b>CloudViewer App</b></sub>
</td>
<td width="50%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/gifs/getting_started_ml_visualizer.gif" width="100%">
<br><sub><b>CloudViewer-ML</b></sub>
</td>
</tr>
</table>

**CloudViewer App** — A standalone 3D viewer app available on Ubuntu and Windows.
Please stay tuned for MacOS. Download from the
[release page](https://github.com/Asher-1/ACloudViewer/releases).

**CloudViewer-ML** — An extension of CloudViewer for 3D machine learning tasks. It builds on
top of the CloudViewer core library and extends it with machine learning tools for
3D data processing. To try it out, install CloudViewer with PyTorch or TensorFlow and check out
[CloudViewer-ML](https://github.com/Asher-1/CloudViewer-ML).

Compilation
-----------

Supported OS: Windows, Linux, and Mac OS X

Refer to the [BUILD.md file](BUILD.md) for detailed build instructions.

Online compilation guides:
- [Build from Source Guide](https://asher-1.github.io/ACloudViewer/documentation/getting_started/build_from_source.html)

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

If you find ACloudViewer useful, please consider supporting its development:

**💰 Financial Support:**

<div align="center">
  <a href="https://asher-1.github.io/ACloudViewer/donation.html">
    <img src="https://img.shields.io/badge/💝_Support_Development-Donate_Now-ff69b4?style=for-the-badge&logo=wechat" alt="Donate">
  </a>
  <p><em>Click to view donation options - Thank you for your support! 🙏</em></p>
</div>

**🌟 Other Ways to Support:**

- ⭐ Star the project on [GitHub](https://github.com/Asher-1/ACloudViewer)
- 🐛 Report bugs and suggest features
- 📝 Contribute code or documentation
- 📢 Share ACloudViewer with others

For more information, see our [Support page](https://asher-1.github.io/ACloudViewer/documentation/resources/support.html).

Thanks for your support!
