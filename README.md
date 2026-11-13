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
* **AICore AI plugins** — depth, feature matching, detection / segmentation, background removal, and 3D Gaussian splats & meshes via GGUF (no Python runtime)
* **3D Gaussian Splatting** real-time rendering and novel view synthesis (SIBR plugin), plus training from photos / video (Spirula Studio integration)
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

A code-level architecture overview — module map, CMake-verified dependency graph, and key data flows — is in [ARCHITECTURE.md](https://github.com/Asher-1/ACloudViewer/blob/main/ARCHITECTURE.md).

---

## AI Agent Integration

ACloudViewer can be controlled by AI agents — [OpenClaw](https://github.com/openclaw),
[Cursor](https://cursor.sh), [Claude Code](https://docs.anthropic.com/en/docs/claude-code),
or any MCP-compatible tool — through a unified integration layer with three interfaces:

| Interface | Protocol | Typical Use |
|-----------|----------|-------------|
| **JSON-RPC Plugin** | WebSocket `ws://127.0.0.1:6001` | Real-time GUI control — load files, run algorithms, capture screenshots from agent code |
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
| **187 MCP tools** | File I/O, cloud/mesh processing, scalar fields, normals, PCV, Compass, SRA, Colmap reconstruction, view control, scene management |
| **89 JSON-RPC methods** | Full GUI automation — load, transform, filter, PCV ambient occlusion, screenshot, export |
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
  <img width="640" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSIBR/images/SIBR_viewer.png">
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

### Spirula Studio — train 3DGS from photos & video

<p align="center">
  <img width="49%" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSpirula/images/spirula_indoor_reconstruction.jpg">
  <img width="49%" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSpirula/images/spirula_outdoor_scene.jpg">
</p>
<p align="center">
  <img width="49%" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSpirula/images/spirula_object_scan.jpg">
  <img width="49%" src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSpirula/images/spirula_360_reconstruction.jpg">
</p>
<p align="center">
<sub>Reconstruction examples from the <a href="https://superspl.at/explore/software/spirula-studio">SuperSplat Spirula Studio Gallery</a> — <b>Ceramic Studio</b> (~50 photos, 1M splats) · <b>Parish Church of Saint Maurice</b> (746 photos, 9.97M splats) · <b>Infinity Octo Art MK</b> (1.99M splats) · <b>Panorama Torunia</b> (533 MB scene, 14.9M splats, native 360° support)</sub>
</p>

The **qSpirula** plugin bundles the upstream
[Spirula Studio](https://github.com/harry7557558/spirula-studio) trainer
(photos / 360° video → splat → textured mesh, GPLv3) as a separate
`spirula` executable next to the application:

* **Launch Spirula Studio** — the official upstream GUI, detached; training, editing, rendering, meshing and device selection run in the official UI
* **Run Headless Training…** — `spirula train <dataset>` with live stdout logs, graceful cancel (SIGTERM → upstream checkpoints) and device selection
* **Import Result to DB Tree…** — `splat.ply` → colored `ccPointCloud` with `Opacity`/`Scale` scalar fields (SH0 → RGB, low-opacity gaussians pruned); `mesh.ply` → `ccMesh`

Enable with `-DBUILD_SPIRULA=ON -DPLUGIN_STANDARD_QSPIRULA=ON`.
See the [qSpirula plugin documentation](https://github.com/Asher-1/ACloudViewer/blob/main/plugins/core/Standard/qSpirula/README.md)
and the [qSpirula user guide](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSpirula.md) for details.

---

## AICore AI Plugins — Depth, Detection, Matching & 3D Generation

Twenty-two GUI plugins share one native inference library — **`libAICore.so`**, whose ggml runtime is vendored from [llama.cpp](https://github.com/ggml-org/llama.cpp). Run quantized **GGUF** models on **CUDA / Vulkan (Linux/Windows) / Metal (macOS) / CPU** with **no Python or PyTorch** at runtime. Results land directly in the DB tree and plug into reconstruction, COLMAP, and SIBR workflows.

**Perception — depth · lines · matching · faces · text**

| Plugin | Task | Model | Standout | CMake |
|--------|------|-------|----------|-------|
| **[qDA3](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDA3.md)** | Monocular & multi-view depth, camera pose | Depth Anything V3 GGUF | Single-image depth cloud in one click; automatic-reconstruction integration | `PLUGIN_STANDARD_QDA3` |
| **[qDeepLSD](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDeepLSD.md)** | Line-segment / wireframe extraction | DeepLSD wireframe GGUF | AFM + LSD lines on photos | `PLUGIN_STANDARD_QDEEPLSD` |
| **[qLightGlue](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLightGlue.md)** | Sparse feature matching | SIFT/ALIKED LightGlue GGUF | 300+ matches in **< 1 s** on GPU | `PLUGIN_STANDARD_QLIGHTGLUE` |
| **[qFaceDetect](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFaceDetect.md)** | Face detect / analyze / verify | face-detect.cpp GGUF packs | SCRFD + ArcFace in one dialog; live multi-face video | `PLUGIN_STANDARD_QFACEDETECT` |
| **[qOCR](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qOCR.md)** | Text / layout / table recognition | PP-OCRv4 GGUF (EN / JA / CH+EN recipes) | SLANet layout + table grids overlaid PP-Structure style | `PLUGIN_STANDARD_QOCR` |

**Detection, segmentation & open-vocabulary understanding**

| Plugin | Task | Model | Standout | CMake |
|--------|------|-------|----------|-------|
| **[qYOLO](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qYOLO.md)** | Detect / seg / depth / pose / OBB / classify + open-vocab | YOLOv8 / YOLO26 GGUF (63 models) | 9 task families incl. World / YOLOE prompts | `PLUGIN_STANDARD_QYOLO` |
| **[qSAM3](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3.md)** | Promptable segmentation + video tracking | SAM2 / 2.1 / 3 GGUF (39 models) | Text / box / point prompts, video tracking | `PLUGIN_STANDARD_QSAM3` |
| **[qRFDetr](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRFDetr.md)** | Detection & instance masks | RF-DETR GGUF (44 models) | 13 ms GPU latency + class allowlist | `PLUGIN_STANDARD_QRFDETR` |
| **[qRMBG](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRMBG.md)** | Background removal | RMBG-2.0 (BiRefNet-Swin-L) GGUF | Transparent RGBA + raw alpha matte | `PLUGIN_STANDARD_QRMBG` |
| **[qRAM](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRAM.md)** | Image tagging & captioning | RAM / RAM++ / Tag2Text GGUF | All logits + thresholds, JSON/CSV/text export | `PLUGIN_STANDARD_QRAM` |
| **[qRelateAnything](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRelateAnything.md)** | Scene relation graphs | RelSGG ViT / ConvNeXt GGUF (24 models) | Subject–predicate–object edges over tracked objects (image & video) | `PLUGIN_STANDARD_QRELATEANYTHING` |
| **[qGKD](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qGKD.md)** | Open-world keypoint detection | GKDT-L GGUF (ECCV 2026) | Text / visual 1-shot / multimodal / multi-object prompts | `PLUGIN_STANDARD_QGKD` |
| **[qVLXSeek](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qVLXSeek.md)** | Open-vocab grounding & region QA | VLX-Seek 1.5-10B GGUF | Detection, counting, VQA, region OCR via region retrieval | `PLUGIN_STANDARD_QVLXSEEK` |

**3D generation & reconstruction**

| Plugin | Task | Model | Standout | CMake |
|--------|------|-------|----------|-------|
| **[qFreeSplatter](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFreeSplatter.md)** | Uncalibrated photos → 3D Gaussian splats | FreeSplatter GGUF | **2 photos** → 3D scene + SIBR PLY; guided Face Capture | `PLUGIN_STANDARD_QFREESPLATTER` |
| **[qTrellis](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qTrellis.md)** | Single image → 3D mesh with PBR | TRELLIS.2 GGUF (DINOv3 + flow DiTs) | PBR-textured GLB in one click | `PLUGIN_STANDARD_QTRELLIS` |
| **[qInstantMesh](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qInstantMesh.md)** | Single image → textured 3D mesh | InstantMesh GGUF (14 files) | Zero123++ 6-view diffusion → FlexiCubes mesh + UV atlas | `PLUGIN_STANDARD_QINSTANTMESH` |
| **[qSAM3D](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3D.md)** | Single image → 3D object | SAM 3D Objects GGUF | Gaussian splat PLY + FlexiCubes mesh | `PLUGIN_STANDARD_QSAM3D` |
| **[qQuerySplat](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qQuerySplat.md)** | 2–16 unposed photos → 3DGS scene | QuerySplat + VGGT-Omega GGUF | 524 288 gaussians feed-forward, optional TTO | `PLUGIN_STANDARD_QQUERYSPLAT` |
| **[qMapAnything](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qMapAnything.md)** | Multi-view metric reconstruction | VGGT-Omega / MapAnything / Pi3 / VGGT-1B / DUSt3R GGUF | 6 model variants, ETH3D benchmark anchoring | `PLUGIN_STANDARD_QMAPANYTHING` |
| **[qLingbotMap](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLingbotMap.md)** | Streaming RGB-D reconstruction | LingBot-Map GCT GGUF | Live per-frame map + camera frustums, native sky masking | `PLUGIN_STANDARD_QLINGBOTMAP` |
| **[qOpenPCDet](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qOpenPCDet.md)** | LiDAR 3D object detection | OpenPCDet GGUF (54 models) | PointPillars / SECOND / PV-RCNN on KITTI + nuScenes | `PLUGIN_STANDARD_QOPENPCDET` |

**Annotation workbench**

| Plugin | Task | Model | Standout | CMake |
|--------|------|-------|----------|-------|
| **[qAnyLabeling](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qAnyLabeling.md)** | AI-assisted image & 3D annotation | AICore catalogs (YOLO, SAM3, RF-DETR, …) | X-AnyLabeling-style workbench, video tracking + 3D cuboids, fully local | `PLUGIN_STANDARD_QANYLABELING` |

### Gallery

<table>
<tr>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qDA3/images/qDA3.png" width="100%">
<br><sub><b>Depth Anything V3</b> — depth maps &amp; 3D unprojection from a single photo</sub>
</td>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qLightGlue/images/qLightGlue.png" width="100%">
<br><sub><b>LightGlue</b> — SIFT / ALIKED matching with live visualization</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qFreeSplatter/images/qFreeSplatter.png" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qFreeSplatter/images/qFreeSplatter_video.png" width="49%"></div>
<br><sub><b>qFreeSplatter</b> — sparse-view 3D Gaussian reconstruction and guided multi-view Face Capture</sub>
</td>
</tr>
<tr>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qFaceDetect/images/qFaceDetect.png" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qFaceDetect/images/qFaceDetect_video.png" width="49%"></div>
<br><sub><b>qFaceDetect</b> — registry authentication, face recognition, and live multi-face video</sub>
</td>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qDeepLSD/images/qDeepLSD.png" width="100%">
<br><sub><b>qDeepLSD</b> — AFM + LSD line-segment extraction on photos</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qYOLO/images/yolo-seg.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qYOLO/images/yolo-pose.jpg" width="49%"></div>
<br><sub><b>qYOLO</b> — detection, segmentation, metric depth, pose, OBB, classification and open-vocab World / YOLOE prompts</sub>
</td>
</tr>
<tr>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qRFDetr/images/qRFDetr.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qRFDetr/images/qRFDetr_video.jpg" width="49%"></div>
<br><sub><b>qRFDetr</b> — real-time RF-DETR detection &amp; instance masks with per-class allowlist</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qRMBG/images/qRMBG.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qRMBG/images/qRMBG_video.jpg" width="49%"></div>
<br><sub><b>qRMBG</b> — one-click transparent background removal (image &amp; live video)</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSAM3/images/qSam3_full.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSAM3/images/qSam3_visual.jpg" width="49%"></div>
<br><sub><b>qSAM3</b> — point / box / text prompts and video object tracking (SAM 2 / 2.1 / 3)</sub>
</td>
</tr>
<tr>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qTrellis/images/qTrellis_f16_1024_pbr.png" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qTrellis/images/qTrellis_q8_512_pbr.png" width="49%"></div>
<br><sub><b>qTrellis</b> — single image → PBR-textured 3D mesh &amp; GLB (TRELLIS.2)</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qGKD/images/text_prompts_with_chair_skeleton.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qGKD/images/multi-objects.jpg" width="49%"></div>
<br><sub><b>qGKD</b> — open-world keypoint detection: text, visual 1-shot, multimodal and multi-object prompts (GKDT-L)</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qOCR/images/ocr_test1.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qOCR/images/table_test1.jpg" width="49%"></div>
<br><sub><b>qOCR</b> — PP-OCRv4 text recognition with SLANet layout &amp; table reconstruction</sub>
</td>
</tr>
<tr>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qInstantMesh/images/instantmesh_test.jpg" width="100%">
<br><sub><b>qInstantMesh</b> — single image → textured 3D mesh (Zero123++ diffusion + FlexiCubes)</sub>
</td>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qSAM3D/images/sam3_seg_with_prompt.jpg" width="100%">
<br><sub><b>qSAM3D</b> — SAM 3D Objects: one photo → Gaussian splat PLY + FlexiCubes mesh</sub>
</td>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qQuerySplat/images/querysplat_test.jpg" width="100%">
<br><sub><b>qQuerySplat</b> — 2–16 unposed photos → a 3D Gaussian Splatting scene (524 288 gaussians)</sub>
</td>
</tr>
<tr>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qMapAnything/images/courtyard_reconstruction.png" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qMapAnything/images/relief_reconstruction.png" width="49%"></div>
<br><sub><b>qMapAnything</b> — multi-view feed-forward metric reconstruction (VGGT-Omega &amp; five more families)</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qLingbotMap/images/courthouse_reconstruction.png" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qLingbotMap/images/loop_reconstruction.png" width="49%"></div>
<br><sub><b>qLingbotMap</b> — streaming RGB-D 3D reconstruction from image folders / video</sub>
</td>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qOpenPCDet/images/pointpillar_test.jpg" width="100%">
<br><sub><b>qOpenPCDet</b> — LiDAR 3D detection: PointPillars / SECOND / PV-RCNN on KITTI + nuScenes</sub>
</td>
</tr>
<tr>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qRAM/images/ram_test.jpg" width="100%">
<br><sub><b>qRAM</b> — Recognize Anything / RAM++ / Tag2Text image tagging &amp; captioning</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qRelateAnything/images/relate_anything_test.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qRelateAnything/images/relate_anything_video_test.jpg" width="49%"></div>
<br><sub><b>qRelateAnything</b> — scene relation graphs: subject–predicate–object edges (image &amp; video)</sub>
</td>
<td width="33%" align="center">
<div style="display:flex;gap:6px;"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qVLXSeek/images/detection.jpg" width="49%"><img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qVLXSeek/images/region_ocr.jpg" width="49%"></div>
<br><sub><b>qVLXSeek</b> — open-vocabulary grounding: detection, counting, VQA, region OCR (VLX-Seek 1.5-10B)</sub>
</td>
</tr>
<tr>
<td width="33%" align="center">
<img src="https://raw.githubusercontent.com/Asher-1/ACloudViewer/main/plugins/core/Standard/qAnyLabeling/docs/images/overview.png" width="100%">
<br><sub><b>qAnyLabeling</b> — X-AnyLabeling-style annotation workbench with fully local AI assist</sub>
</td>
</tr>
</table>

**Why AICore?**

* **Native C++ end-to-end** — GUI, automatic reconstruction, and COLMAP pipelines without a Python stack
* **Compact GGUF weights** — e.g. DA3 Base ~142 MB, LightGlue SIFT matcher ~22 MB; four matching families with one-click download in the dialog
* **Multi-backend GPU** — Auto picks CUDA → Vulkan → CPU (Linux/Windows) or Metal → CPU (macOS)
* **DB-tree integration** — depth clouds, match lines, Gaussian PLY, and camera frustums appear as first-class entities

```bash
cmake -B build_app \
  -DBUILD_GUI=ON \
  -DAICore_ENABLED=ON \
  -DPLUGIN_STANDARD_QDA3=ON \
  -DPLUGIN_STANDARD_QDEEPLSD=ON \
  -DPLUGIN_STANDARD_QFACEDETECT=ON \
  -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
  -DPLUGIN_STANDARD_QFREESPLATTER=ON \
  -DPLUGIN_STANDARD_QSAM3=ON \
  -DPLUGIN_STANDARD_QSAM3D=ON \
  -DPLUGIN_STANDARD_QYOLO=ON \
  -DPLUGIN_STANDARD_QTRELLIS=ON \
  -DPLUGIN_STANDARD_QRFDETR=ON \
  -DPLUGIN_STANDARD_QLINGBOTMAP=ON \
  -DPLUGIN_STANDARD_QRMBG=ON \
  -DPLUGIN_STANDARD_QGKD=ON \
  -DPLUGIN_STANDARD_QOCR=ON \
  -DPLUGIN_STANDARD_QRAM=ON \
  -DPLUGIN_STANDARD_QRELATEANYTHING=ON \
  -DPLUGIN_STANDARD_QANYLABELING=ON \
  -DPLUGIN_STANDARD_QMAPANYTHING=ON \
  -DPLUGIN_STANDARD_QOPENPCDET=ON \
  -DPLUGIN_STANDARD_QINSTANTMESH=ON \
  -DPLUGIN_STANDARD_QVLXSEEK=ON \
  -DPLUGIN_STANDARD_QQUERYSPLAT=ON \
  -DBUILD_RECONSTRUCTION=ON \
  -DPLUGIN_STANDARD_QSIBR=ON \
  .

cmake --build build_app --target ACloudViewer -j$(nproc)
```

User guides: [AICore plugins overview](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/README.md) · [qDA3](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDA3.md) · [qDeepLSD](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qDeepLSD.md) · [qFaceDetect](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFaceDetect.md) · [qLightGlue](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLightGlue.md) · [qFreeSplatter](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qFreeSplatter.md) · [qYOLO](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qYOLO.md) · [qSAM3](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3.md) · [qTrellis](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qTrellis.md) · [qRFDetr](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRFDetr.md) · [qRMBG](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRMBG.md) · [qGKD](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qGKD.md) · [qInstantMesh](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qInstantMesh.md) · [qLingbotMap](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qLingbotMap.md) · [qMapAnything](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qMapAnything.md) · [qOCR](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qOCR.md) · [qOpenPCDet](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qOpenPCDet.md) · [qQuerySplat](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qQuerySplat.md) · [qRAM](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRAM.md) · [qRelateAnything](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qRelateAnything.md) · [qSAM3D](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSAM3D.md) · [qVLXSeek](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qVLXSeek.md) · [qAnyLabeling](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qAnyLabeling.md) · [qSpirula](https://github.com/Asher-1/ACloudViewer/blob/main/docs/guides/plugins/qSpirula.md)

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
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Asher-1/ACloudViewer&type=date&theme=dark&legend=top-left&sealed_token=xOg1m_uF2wlk5OvfhJb3qcLLyip3ZfuKuMANnVmW9FyJaokgswH8NdFi-maimTqxb9Fg3Axe5WwmI3qIFsR6b4buaQEcv168M7N4-pCzY2IqwVkGxp9WhmFmFNdHRbdrSvp63ecnx-gixg81ShqHbSgWxc_wSpBOy-3fa0HcbMLXc15jUCgdXb8gwEHO" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Asher-1/ACloudViewer&type=date&legend=top-left&sealed_token=xOg1m_uF2wlk5OvfhJb3qcLLyip3ZfuKuMANnVmW9FyJaokgswH8NdFi-maimTqxb9Fg3Axe5WwmI3qIFsR6b4buaQEcv168M7N4-pCzY2IqwVkGxp9WhmFmFNdHRbdrSvp63ecnx-gixg81ShqHbSgWxc_wSpBOy-3fa0HcbMLXc15jUCgdXb8gwEHO" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Asher-1/ACloudViewer&type=date&legend=top-left&sealed_token=xOg1m_uF2wlk5OvfhJb3qcLLyip3ZfuKuMANnVmW9FyJaokgswH8NdFi-maimTqxb9Fg3Axe5WwmI3qIFsR6b4buaQEcv168M7N4-pCzY2IqwVkGxp9WhmFmFNdHRbdrSvp63ecnx-gixg81ShqHbSgWxc_wSpBOy-3fa0HcbMLXc15jUCgdXb8gwEHO" />
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
