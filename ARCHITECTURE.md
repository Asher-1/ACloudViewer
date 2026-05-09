# ACloudViewer Architecture

A Modern System for 3D Data Processing — open-source library for rapid development
of software dealing with 3D data, based on CloudCompare, Open3D, ParaView, and COLMAP.

## System Overview

ACloudViewer is a desktop application built with **C++17**, using **Qt** for the GUI,
**VTK** for 3D rendering (conditional via `USE_VTK_BACKEND`), **PCL** for point cloud
algorithms, and **CUDA** for GPU acceleration. The codebase spans ~6,400 source files
organized into 11 architectural layers.

```
┌─────────────────────────────────────────────────────────┐
│                    Qt GUI Application                    │
│           (MainWindow, dialogs, tools, DB tree)          │
├──────────────────────┬──────────────────────────────────┤
│  VTK Visualization   │   Plugin System (35+ plugins)    │
│   Engine (vtkGLView, │   (I/O, algorithms, PCL, SIBR    │
│   VtkDisplayTools)   │    Gaussian Splatting, Python)    │
├──────────────────────┼──────────────────────────────────┤
│     Spatial Object Database (CV_db)                      │
│  (ccHObject scene graph, sensors, scalar fields, draw)   │
├──────────────────────┬──────────────────────────────────┤
│  CVCoreLib (core/)   │   Geometry Core (cloudViewer/)   │
│  (octree, distances, │   (Tensor, Open3D, CUDA,         │
│   sampling, CC types)│    t::geometry, pipelines)        │
├──────────────────────┼──────────────────────────────────┤
│  Scene Reconstruction│   Viewer Infrastructure & I/O    │
│  (COLMAP SfM/MVS)    │   (CVViewer, CVAppCommon, CV_io) │
├──────────────────────┴──────────────────────────────────┤
│  Agent Integration (JSON-RPC WebSocket, MCP, CLI)        │
├──────────────────────────────────────────────────────────┤
│  Python Ecosystem (pybind11, examples, ML integration)   │
├──────────────────────────────────────────────────────────┤
│  Build System & CI/CD (CMake, GitHub Actions, Docker)    │
└─────────────────────────────────────────────────────────┘
```

## Repository Map

```
ACloudViewer/
├── app/                    # Qt desktop application (MainWindow, dialogs, tools)
├── core/                   # CVCoreLib — scalar core (octrees, distances, sampling)
├── libs/
│   ├── cloudViewer/        # Open3D-derived geometry & tensor library
│   ├── CV_db/              # Spatial object database (scene graph, entities)
│   ├── VtkEngine/          # VTK rendering backend (conditional: USE_VTK_BACKEND)
│   ├── Reconstruction/     # COLMAP integration (conditional: BUILD_RECONSTRUCTION)
│   ├── CVViewer/           # Viewer application scaffolding
│   ├── CVAppCommon/        # Shared dialogs, plugin manager, themes
│   ├── CV_io/              # File format I/O drivers
│   ├── CVPluginAPI/        # Plugin API headers
│   ├── CVPluginStub/       # Plugin stub implementations
│   └── Python/             # pybind11 bindings (conditional: BUILD_PYTHON_MODULE)
├── plugins/                # 35+ loadable plugins (I/O, algorithms, PCL, SIBR)
├── agent-integration/      # AI agent control layer (JSON-RPC, MCP, CLI)
├── examples/               # C++ and Python example scripts
├── tests/                  # Test suites
├── 3rdparty/               # Vendored third-party dependencies
├── cmake/                  # CMake modules and version config
├── docker/                 # Docker build environments
├── docs/                   # Documentation and user guides
└── .github/workflows/      # CI/CD pipelines (Ubuntu, macOS, Windows, CUDA)
```

## Entry Points

- **Desktop App:** `app/main.cpp` → `MainWindow` (Qt GUI)
- **CLI:** `app/ecvCommandLineParser.cpp` (batch processing)
- **Python:** `libs/Python/pybind/cloudViewer_pybind.cpp`
- **Agent Integration:** `agent-integration/` — three interfaces:
  - **JSON-RPC Plugin:** WebSocket `ws://localhost:6001` for real-time GUI control
  - **MCP Server:** Model Context Protocol (stdio) for Cursor/Claude Code/OpenClaw
  - **CLI Harness:** Click CLI + REPL for shell scripts and headless batch processing
  - See `agent-integration/README.md` and `agent-integration/docs/` for full API docs

## Module Reference

### 1. Qt GUI Application (`app/`)

The main desktop application with 257 C++ files.

| Component | Files | Purpose |
|-----------|-------|---------|
| `MainWindow` | `MainWindow.h/cpp` (14k lines) | Central hub: menus, toolbar, view management, action routing |
| `db_tree/` | `ecvDBRoot`, `ecvPropertiesTreeDelegate` | Scene tree model/view with property editing |
| Dialogs | `ecv*Dlg.cpp` (30+ dialogs) | Registration, segmentation, color, normals, etc. |
| Tools | `ecv*Tool.cpp` | Crop, segmentation, annotation, transformation |
| Views | `ecvMultiViewWidget`, `ecvSpreadSheetView` | Multi-view layout management, data tables |
| Commands | `ecvCommandLineCommands` | CLI command implementations |

### 2. VTK Visualization Engine (`libs/VtkEngine/`)

332 files implementing the VTK-based rendering backend.

| Subsystem | Key Classes | Purpose |
|-----------|-------------|---------|
| Visualization | `vtkGLView`, `VtkDisplayTools`, `VtkVis` | Per-window 3D view, CC→VTK drawing bridge |
| Views | `vtkChartView` | Line/bar/histogram/box/parallel/scatter charts |
| Widgets | `QVTKWidgetCustom`, `ScaleBarWidget` | Custom QVTK widget with overlays |
| Camera | `VtkCameraLink` | Pairwise named camera synchronization |
| Interaction | `vtkCustomInteractorStyle` | ParaView-style trackball interaction |
| Converters | `Cc2Vtk`, `Vtk2Cc` | ccPointCloud/ccMesh ↔ vtkPolyData |
| Tools | Annotation, Measurement, Selection, Transform | Interactive VTK-based editing tools |

**Key Architecture Pattern:** Dual state model — CC-style `ecvViewContext`/`CC_DRAW_CONTEXT`
is synchronized with VTK camera/renderer via `syncVtkCameraToContext()`. Entity `draw()` calls
flow through `VtkDisplayTools` which converts CC geometry to VTK actors via `Cc2Vtk`.

### 3. Spatial Object Database (`libs/CV_db/`)

242 files defining the scene graph and entity model.

**Class Hierarchy:**
```
ccSerializableObject
  └── ccObject
        └── ccHObject (+ ccDrawableObject)
              ├── ccShiftedObject
              │     ├── ccGenericPointCloud → ccPointCloud
              │     ├── ccGenericMesh → ccMesh, ccSubMesh
              │     └── ccPolyline
              ├── ccSensor
              │     ├── ccCameraSensor (frustum, dirty-flag optimized)
              │     └── ccGBLSensor (laser scanner)
              ├── cc2DLabel (+ ccInteractor)
              ├── ccFacet, ccImage, ccClipBox
              └── ecvOrientedBBox, ccBBox
```

**Display Interface:** `ecvGenericGLDisplay` (per-window abstraction) → implemented by
`vtkGLView`. `ecvViewManager` is the singleton hub managing all registered views.

**Draw Pipeline:** Entity `draw(CC_DRAW_CONTEXT&)` → `drawMeOnly()` → `context.display->draw(ctx, this)` → VTK backend.

### 4. CVCoreLib (`core/`)

The foundational scalar/spatial library inherited from CloudCompare.

| Component | Purpose |
|-----------|---------|
| `CCCoreLib/` | Octree structures, distance computation, sampling algorithms |
| Scalar types | `ScalarType`, `ScalarField`, `CCVector3`, `CCVector2` |
| Spatial | Nearest neighbor search, Chamfer distance, Hausdorff |
| Math | Fitting (plane/sphere/cylinder), Delaunay triangulation |

Build order: `core/` is compiled before `libs/` (`add_subdirectory(core)` in root CMakeLists).

### 5. Geometry Core & Algorithms (`libs/cloudViewer/`)

~1,050 files — Open3D-derived geometry and tensor library.

| Area | Description |
|------|-------------|
| `core/` | Tensor, Device (CPU/CUDA), dtype dispatch, spatial hash maps, NNS |
| `t/geometry/` | Device-aware geometries: PointCloud, TriangleMesh, LineSet, Image |
| `pipelines/` | Legacy Eigen-based: registration, TSDF, color map, odometry |
| `t/pipelines/` | Tensor-based: ICP, RGB-D odometry, SLAC/SLAM |
| `ml/` | TensorFlow/PyTorch custom ops |
| `geometry/` | **Shim headers** mapping `cloudViewer::geometry::*` → CC types |

**Dual API:** `cloudViewer::geometry::PointCloud` aliases `ccPointCloud` (CloudCompare);
`cloudViewer::t::geometry::PointCloud` is the tensor-based GPU-ready implementation.

### 6. Scene Reconstruction (`libs/Reconstruction/`)

~750 files — COLMAP-oriented photogrammetric reconstruction (conditional: `BUILD_RECONSTRUCTION`).
GUI integration shell at `app/reconstruction/`.

### 7. Viewer Infrastructure (`libs/CVViewer/`, `libs/CVAppCommon/`, `libs/CV_io/`)

439 files — shared utilities and file format drivers.

| Module | Purpose |
|--------|---------|
| `CVViewer/` | Viewer scaffolding, application base |
| `CVAppCommon/` | Shared dialogs, plugin manager, themes |
| `CV_io/` | Format drivers: PLY, LAS, PCD, E57, FBX, OBJ, STL, etc. |

**I/O Pipeline:** `FileIOFilter::LoadFromFile()` → filter registry → format-specific
`loadFile()` → returns `ccHObject*` tree.

### 8. Plugins & Extension API (`libs/CVPluginAPI/`, `libs/CVPluginStub/`, `plugins/`)

Extensible plugin system with 35+ plugins.

**Plugin Types:**
- `ECV_STD_PLUGIN` — GUI algorithms with menu actions
- `ECV_IO_FILTER_PLUGIN` — File format readers/writers
- `ECV_PCL_ALGORITHM_PLUGIN` — PCL-wrapped algorithms

**Available Plugins:**

| Category | Plugins |
|----------|---------|
| I/O | qCoreIO, qAdditionalIO, qMeshIO, qCSVMatrixIO, qE57IO, qFBXIO, qLASIO, qLASFWFIO, qPDALIO, qPhotoscanIO, qRDBIO, qStepCADImport, qDracoIO |
| Algorithms | qCSF (ground filter), qM3C2, qRANSAC_SD (shapes), qPoissonRecon, qHoughNormals, qPCV |
| Classification | q3DMASC, qCANUPO, qCloudLayers |
| Structural | qCompass, qCork (CSG), qFacets, qG3Point, qMPlane, qSRA |
| Segmentation | qColorimetricSegmenter, qMasonry (qManualSeg/qAutoSeg) |
| Forestry | qTreeIso, qVoxFall |
| Gaussian Splatting | **qSIBR** (CUDA real-time rendering, remote viewer) |
| Integration | qPCL (PCL algorithms), qPythonRuntime, qJSonRPCPlugin |
| Visualization | qAnimation |
| Examples | ExamplePlugin, ExampleIOPlugin, ExampleGLPlugin (`plugins/example/`) |

### 9. Agent Integration (`agent-integration/`)

AI agent control layer with three interfaces:

| Interface | Protocol | Files |
|-----------|----------|-------|
| JSON-RPC Plugin | WebSocket `ws://localhost:6001` | `cli/`, `examples/` |
| MCP Server | Model Context Protocol (stdio) | `mcp/` |
| CLI Harness | Click CLI + REPL | `cli/` |

See `agent-integration/docs/JSON-RPC-API.md` and `agent-integration/docs/COMMAND-MAPPING.md`
for complete API reference. CI pipeline: `.github/workflows/agent-integration.yml`.

### 10. Python Ecosystem (`libs/Python/`, `examples/`)

pybind11 bindings and example scripts.

| Subsystem | Purpose |
|-----------|---------|
| `pybind/geometry/` | Point cloud, mesh, line set bindings |
| `pybind/t/` | Tensor-based geometry bindings |
| `pybind/pipelines/` | Registration, TSDF, odometry |
| `pybind/reconstruction/` | COLMAP reconstruction bindings |
| `pybind/visualization/` | Viewer, WebRTC server |
| `pybind/ml/` | PyTorch/TensorFlow integration |
| `examples/Python/` | Tutorial scripts |
| `examples/Cpp/` | C++ usage examples |

### 11. Build System (`cmake/`, `.github/workflows/`, `docker/`)

CMake-based build with conditional feature flags and CI/CD.

**Conditional Build Flags:**

| Flag | Default | Controls |
|------|---------|----------|
| `USE_VTK_BACKEND` | ON | VTK rendering engine (`libs/VtkEngine/`) |
| `BUILD_RECONSTRUCTION` | ON | COLMAP integration (`libs/Reconstruction/`) |
| `BUILD_PYTHON_MODULE` | ON | Python bindings (`libs/Python/`) |
| `BUILD_GUI` | ON | Desktop GUI application (`app/`) |

**CI/CD Pipelines:** Ubuntu, macOS, Windows, CUDA wheel builds, WebRTC, CodeQL,
agent-integration, documentation.

## Key Data Flows

### File Open → Render
```
MainWindow::doActionOpenFile()
  → FileIOFilter::LoadFromFile(filename, filter)
    → PlyFilter::loadFile() → ccPointCloud / ccMesh
  → MainWindow::addToDB(ccHObject*)
    → ccDBRoot::addElement() → scene tree
    → ecvViewManager::associateToActiveView() → setDisplay_recursive()
  → refreshAll() → ecvRedrawScope
    → vtkGLView::redraw()
      → m_globalDBRoot->draw(CC_DRAW_CONTEXT)
        → ccPointCloud::drawMeOnly() → context.display->draw()
          → VtkDisplayTools::draw()
            → VtkVis::drawPointCloud() → Cc2Vtk::PointCloudToPolyData()
              → CreateActorFromVTKDataSet() → addActorToRenderer()
      → vtkRenderWindow::Render()
```

### Sensor Frustum Drawing
```
ccCameraSensor::drawMeOnly(CC_DRAW_CONTEXT)
  → Check m_geometryDirty || transformChanged (memcmp cache)
  → If dirty: updateData() → computeFrustumCorners() → rebuild LineSet
  → context.display->draw(ctx, this)
    → VtkDisplayTools::drawSensor()
      → VtkVis::drawSensor() → CreateCameraSensor(LineSet → vtkPolyData)
```

### Chart View Update
```
ecvViewManager::entitySelectionChanged
  → vtkChartView::onEntitySelectionChanged()
    → setEntity(ccPointCloud*)
      → populate field list from scalar fields
      → rebuildChart()
        → ccScalarField::getValue() → vtkTable columns
        → vtkChart::AddPlot() → vtkContextView::Render()
```

### Plugin Loading
```
main.cpp → ccPluginManager::loadPlugins()
  → scan paths → QPluginLoader for each .so/.dll
  → validate info.json metadata
  → ECV_IO_FILTER_PLUGIN → FileIOFilter::Register()
  → ECV_STD_PLUGIN → ccExternalFactory registration
→ ccPluginUIManager::init()
  → setMainAppInterface() for each plugin
  → getActions() → attach to menus/toolbars
→ User clicks → QAction::triggered → plugin handler
```

## Edit Here For...

| Task | Where to Look |
|------|---------------|
| Add new file format | `libs/CV_io/` + `plugins/core/IO/` (implement `FileIOFilter`) |
| Add new algorithm | `plugins/core/Standard/` (implement `ccStdPluginInterface`) |
| Modify 3D rendering | `libs/VtkEngine/Visualization/VtkDisplayTools.cpp` |
| Add new view type | `libs/VtkEngine/VTKExtensions/Views/` + `app/ecvMultiViewWidget.cpp` |
| Modify entity properties | `libs/CV_db/include/ecv*.h` + `app/db_tree/ecvPropertiesTreeDelegate.cpp` |
| Add sensor type | `libs/CV_db/include/ecvSensor.h` → derive (see `ecvCameraSensor`, `ecvGBLSensor`) |
| Modify camera linking | `libs/VtkEngine/Visualization/VtkCameraLink.h/cpp` |
| Add Python binding | `libs/Python/pybind/` (modular: geometry, pipelines, visualization, ml) |
| Modify reconstruction | `libs/Reconstruction/` + `app/reconstruction/` (GUI shell) |
| Add agent command | `agent-integration/cli/` + `agent-integration/docs/COMMAND-MAPPING.md` |
| Modify build system | `CMakeLists.txt` + `cmake/CMakeVersionConfig.cmake` |
| Add CI workflow | `.github/workflows/` |

## Frameworks & Dependencies

| Framework | Version | Purpose |
|-----------|---------|---------|
| Qt | 5.x/6.x | GUI, signals/slots, widgets |
| VTK | 9.x | 3D rendering, charts, interaction |
| PCL | 1.x | Point cloud algorithms |
| CUDA | 11+ | GPU acceleration |
| Eigen | 3.x | Linear algebra |
| OpenCV | 4.x | Image processing |
| COLMAP | - | Scene reconstruction |
| pybind11 | - | Python bindings |
| CMake | 3.19+ | Build system |

## Naming Conventions

The codebase uses a dual naming convention from its CloudCompare heritage:
- **File names** use `ecv` prefix: `ecvHObject.h`, `ecvPointCloud.h`, `ecvCameraSensor.h`
- **Class names** use `cc` prefix: `ccHObject`, `ccPointCloud`, `ccCameraSensor`
- **VtkEngine classes** use `vtk` prefix: `vtkGLView`, `vtkChartView`, `VtkDisplayTools`
- **Plugin manager:** file `ecvPluginManager.h` defines class `ccPluginManager`

## Knowledge Graph

An interactive architecture dashboard is available at `.understand-anything/knowledge-graph.json`.
Run `/understand-dashboard` to launch the interactive explorer.
