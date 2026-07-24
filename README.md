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
    <a href="https://asher-1.github.io/ACloudViewer/#aicore">AICore AI</a> |
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
* **AICore AI plugins** — depth estimation, feature matching, and 3D Gaussian splats via GGUF (no Python runtime)
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

---

## AI Agent Integration

ACloudViewer can be controlled by AI agents — [OpenClaw](https://github.com/openclaw),
[Cursor](https://cursor.sh), [Claude Code](https://docs.anthropic.com/en/docs/claude-code),
or any MCP-compatible tool — through a unified integration layer with three interfaces:

| Interface | Protocol | Typical Use |
|-----------|----------|-------------|
| **JSON-RPC Plugin** | WebSocket `ws://localhost:6001` | Real-time GUI control — load files, run algorithms, capture screenshots from agent code |
| **MCP Server** | Model Context Protocol (stdio) | Native integration with Cursor IDE, Claude Code, OpenClaw |
| **CLI Harness** | Click CLI + REPL | Shell scripts, headless batch processing, CI pipelines |

Browse all available tools on the [CLI-Anything Hub](https://asher-1.github.io/CLI-Anything/).

### Quick start

```bash
# 1. Install the CLI harness (headless or GUI control)
pip install git+https://github.com/Asher-1/CLI-Anything.git#subdirectory=acloudviewer/agent-harness

# 2. Convert point cloud formats (headless — no GUI needed)
cli-anything-acloudviewer --mode headless convert input.ply output.pcd

# 3. Subsample with spatial voxel grid
cli-anything-acloudviewer --mode headless process subsample input.ply -o sub.ply --voxel-size 0.05

# 4. Compute normals
cli-anything-acloudviewer --mode headless process normals input.ply -o normals.ply

# 5. Start MCP server (auto-detects running GUI or falls back to headless)
cli-anything-acloudviewer-mcp --mode auto
```

### What's included

| Component | Scope |
|-----------|-------|
| **178 MCP tools** | File I/O, cloud/mesh processing, scalar fields, normals, PCV, Compass, SRA, Colmap reconstruction, view control, scene management |
| **72 JSON-RPC methods** | Full GUI automation — load, transform, filter, PCV ambient occlusion, screenshot, export |
| **40+ CLI commands** | `convert`, `process` (subsample, normals, PCV, Compass export/refit/P21, SRA, density, curvature, SOR, ICP, …), `view`, `scene`, `colmap` |
| **40+ file formats** | PLY, PCD, LAS/LAZ, E57, FBX, OBJ, STL, DRC, SBF, VTK, ASC, XYZ, CSV, PTS, … |

Headless mode invokes the ACloudViewer binary directly (no Python bindings needed),
supporting all plugin-provided formats (LAS/LAZ, E57, FBX, PCD, Draco).

Enable the JSON-RPC plugin at build time: `-DPLUGIN_STANDARD_QJSONRPC=ON`

Browse all available tools on the [CLI-Anything Hub](https://asher-1.github.io/CLI-Anything/).
See [`agent-integration/`](https://github.com/Asher-1/ACloudViewer/tree/main/agent-integration) for full documentation, MCP tool reference,
and the [unified test suite](https://github.com/Asher-1/ACloudViewer/tree/main/agent-integration/tests) (267 tests across 5 levels).

---

## SIBR Viewer — 3D Gaussian Splatting & Novel View Synthesis

<p align="center">
  <img width="640" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/SIBR_viewer.png">
</p>

ACloudViewer integrates the **SIBR framework** (System for Image-Based Rendering) as a built-in plugin,
enabling real-time **3D Gaussian Splatting** rendering and novel view synthesis directly within the application.

| Viewer | Description |
|--------|-------------|
| **3D Gaussian Splatting** | Real-time CUDA-accelerated rendering of trained 3DGS models (`.ply` splat files) |
| **Remote Gaussian** | Live connection to a running 3DGS training process for real-time monitoring |
| **ULR / ULR v2** | Unstructured Lumigraph Rendering for novel view synthesis |
| **Textured Mesh & Point-Based** | IBR dataset visualization with scene debug overlays |

Key features:
* **Bidirectional interaction** — Select entities in ACloudViewer, auto-detect best viewer, import results back with auto-zoom
* **No separate install** — The SIBR viewers are built as an ACloudViewer plugin, launchable from the GUI toolbar
* **Multi-format ingest** — Load Colmap reconstructions, SIBR datasets, or raw 3DGS `.ply` files

Enable with `-DPLUGIN_STANDARD_QSIBR=ON -DBUILD_CUDA_MODULE=ON` (CUDA optional for non-3DGS viewers).
See [qSIBR plugin documentation](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSIBR/README.md) for details.

---

## AICore AI Plugins — Depth, Matching & 3D Gaussian Splats

Three GUI plugins share one native inference library — **`libAICore.so`** ([ggml](https://github.com/ggml-org/ggml)). Run quantized **GGUF** models on **CUDA / Vulkan / Metal / CPU** with **no Python or PyTorch** at runtime. Results land directly in the DB tree and plug into reconstruction, COLMAP, and SIBR workflows.

| | **qDA3** | **qLightGlue** | **qFreeSplatter** |
|---|----------|----------------|-------------------|
| **Task** | Monocular & multi-view depth, camera pose | Sparse feature matching | Uncalibrated photos → 3D Gaussian splats |
| **Model** | Depth Anything V3 GGUF | SIFT + LightGlue GGUF | FreeSplatter GGUF |
| **Standout** | Single-image depth cloud in one click | 300+ matches in **< 1 s** on GPU | **2 photos** → 3D scene + SIBR PLY |
| **CMake** | `PLUGIN_STANDARD_QDA3` | `PLUGIN_STANDARD_QLIGHTGLUE` | `PLUGIN_STANDARD_QFREESPLATTER` |

<table>
<tr>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/qDA3.png" width="100%">
<br><sub><b>Depth Anything V3</b> — depth maps &amp; 3D unprojection from a single photo</sub>
</td>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/qLightGlue.png" width="100%">
<br><sub><b>LightGlue</b> — sub-second SIFT feature matching with live visualization</sub>
</td>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/docs/images/qFreeSplatter.png" width="100%">
<br><sub><b>FreeSplatter</b> — sparse-view 3D Gaussian reconstruction, optional qSIBR preview</sub>
</td>
</tr>
</table>

**Why AICore?**

* **Native C++ end-to-end** — GUI, automatic reconstruction, and COLMAP pipelines without a Python stack
* **Compact GGUF weights** — e.g. DA3 Base ~142 MB, LightGlue SIFT ~22 MB; one-click download in the dialog
* **Multi-backend GPU** — Auto picks CUDA → Vulkan → CPU (Linux/Windows) or Metal → CPU (macOS)
* **DB-tree integration** — depth clouds, match lines, Gaussian PLY, and camera frustums appear as first-class entities

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDA3=ON \
  -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DBUILD_RECONSTRUCTION=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  .

cmake --build build_app --target ACloudViewer -j$(nproc)
```

User guides: [AICore plugins overview](docs/guides/plugins/README.md) · [qDA3](docs/guides/plugins/qDA3.md) · [qLightGlue](plugins/core/Standard/qLightGlue/README.md) · [qFreeSplatter](docs/guides/plugins/qFreeSplatter.md)

---

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

### Semantic Annotation

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

### Reconstruction & Measurement

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

Refer to the [BUILD.md file](https://github.com/Asher-1/ACloudViewer/blob/main/BUILD.md) for detailed build instructions.

Online compilation guides:
- [Build from Source Guide](https://asher-1.github.io/ACloudViewer/documentation/getting_started/build_from_source.html)

Basically, you have to:

- clone this repository
- install mandatory dependencies (OpenGL, etc.) and optional ones if you really need them
  (mainly to support particular file formats, or for some plugins)
- launch CMake (from the trunk root)
- enjoy!

## Star History

<a href="https://www.star-history.com/?repos=Asher-1%2FACloudViewer&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Asher-1/ACloudViewer&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Asher-1/ACloudViewer&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Asher-1/ACloudViewer&type=date&legend=top-left" />
 </picture>
</a>

## StarMapper

<a href="https://starmapper.bruniaux.com/Asher-1/ACloudViewer">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://starmapper.bruniaux.com/api/map-image/Asher-1/ACloudViewer?theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://starmapper.bruniaux.com/api/map-image/Asher-1/ACloudViewer?theme=light" />
    <img alt="StarMapper" src="https://starmapper.bruniaux.com/api/map-image/Asher-1/ACloudViewer" />
  </picture>
</a>

Contributing to ACloudViewer
----------------------------

If you want to help us improve ACloudViewer or create a new plugin you can start by reading
this [guide](https://github.com/Asher-1/ACloudViewer/blob/main/CONTRIBUTING.md)

Supporting the project
----------------------

If you find ACloudViewer useful, please consider supporting its development:

**💰 Financial Support:**

<div align="center">
  <a href="https://asher-1.github.io/ACloudViewer/donation.html">
    <img src="https://img.shields.io/badge/💝_WeChat_Pay-Donate-09BB07?style=for-the-badge&logo=wechat&logoColor=white" alt="WeChat Pay">
  </a>
  &nbsp;
  <a href="https://www.paypal.com/ncp/payment/EHLN9HQV9U39J">
    <img src="https://img.shields.io/badge/💳_PayPal-Donate-00457C?style=for-the-badge&logo=paypal&logoColor=white" alt="PayPal">
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
