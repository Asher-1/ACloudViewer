# AGENTS.md — ACloudViewer Guide

Reference layout: Use this file when you need a **full-repo map** (build, modules, conventions). For scoped rules, also read `.cursor/rules/*.mdc`.

## Project Overview

ACloudViewer is an open-source **3D point cloud and mesh processing** application and library, descended from CloudCompare with integrations for Open3D, ParaView-style visualization, **COLMAP** reconstruction, VTK rendering, Python bindings, and optional **AI inference** (Depth Anything V3, FreeSplatter via ggml). Primary language: **C++17**. GUI: **Qt 5/6**. Optional **CUDA**, **Vulkan/Metal/OpenCL** (ggml backends), and **Python 3.10+**.

Main deliverables:

| Target | Description |
|--------|-------------|
| **ACloudViewer** | Full Qt GUI (`app/`) |
| **CloudViewer** | Library / lighter viewer build |
| **libAICore.so** | Unified DA3 + FreeSplatter inference (`core/AICore/`) |
| **COLMAP** | Reconstruction stack (`libs/Reconstruction/`, optional) |
| **Python** | `cloudViewer` package via pybind (`libs/Python/`) |
| **Plugins** | Dynamic `.so` / `.dylib` under `plugins/core/` |

Agent control: JSON-RPC WebSocket plugin, MCP server, CLI harness — see `agent-integration/README.md`.

## Directory Structure

| Path | Description |
|------|-------------|
| `CMakeLists.txt` | Root build; options for GUI, CUDA, plugins, AICore, reconstruction |
| `cmake/` | Version config, dependency helpers, print summaries |
| `3rdparty/` | Vendored and fetched deps (ggml, OpenCV, VTK, etc.) |
| `core/` | **CVCoreLib** (octree, algorithms) + **AICore** (DA3, FreeSplatter) |
| `libs/` | Application libraries (see Module Layers) |
| `app/` | **ACloudViewer** GUI: MainWindow, DB tree, reconstruction UI, plugins manager |
| `plugins/` | Qt plugins (`core/Standard/`, `core/IO/`), `cmake/Plugins.cmake` |
| `examples/` | Sample C++ programs |
| `docs/` | Sphinx guides, compiling docs, plugin user guides |
| `docker/` | Container build scripts |
| `agent-integration/` | JSON-RPC / MCP / CLI agent docs and examples |
| `util/` | CI helpers (`ci_utils.sh`, `ci_utils.ps1`) |
| `scripts/` | Platform build scripts |
| `.github/workflows/` | CI: Ubuntu, macOS, Windows, CUDA, docs, agent-integration |
| `BUILD.md` | CMake option table and build recipes |
| `plugins/README.md` | Plugin catalog index |

## Module Dependency Layers (bottom → top)

| Layer | Path | Description |
|-------|------|-------------|
| Third-party | `3rdparty/` | ggml, Eigen, FLANN, zlib, optional OpenCV/FFmpeg |
| Core algorithms | `core/` (`CVCoreLib`) | Point cloud structures, octree, scalar fields, basic processing |
| AI inference | `core/AICore/` | `libAICore.so`: `depth_capi`, `gaussian_capi`, ggml backends |
| Database / entities | `libs/CV_db/` | `ccHObject`, `ccPointCloud`, `ccMesh`, `ecvImage`, DB tree model |
| I/O | `libs/CV_io/` | File readers/writers shared with core |
| Visualization | `libs/VtkEngine/` | VTK/GL pipeline, display tools, LOD |
| App common | `libs/CVAppCommon/` | Shared dialogs, UI widgets |
| Reconstruction | `libs/Reconstruction/` | COLMAP-derived SfM/MVS; `DA3DepthController`, fusion |
| Unified library | `libs/cloudViewer/` | Object libraries assembled into CloudViewer lib |
| Python | `libs/Python/` | pybind11 module |
| Plugin API | `libs/CVPluginAPI/`, `libs/CVPluginStub/` | `ccStdPluginInterface`, stub loader |
| GUI app | `app/` | Main window, properties tree, reconstruction widgets |
| Plugins | `plugins/core/` | qDA3, qFreeSplatter, qManualCalib, qSIBR, I/O filters, … |

## Key Classes & Files

| Class / File | Location | Purpose |
|--------------|----------|---------|
| `ccHObject` | `libs/CV_db/` | Base of DB hierarchy (clouds, meshes, groups, images) |
| `ccPointCloud` | `libs/CV_db/include/ecvPointCloud.h` | Point cloud + scalar fields + colors + octree child |
| `ccMesh` | `libs/CV_db/` | Triangle mesh entity |
| `ecvImage` | `libs/CV_db/include/ecvImage.h` | Raster / depth image in DB; DA3 depth hooks |
| `ecvMainAppInterface` | `libs/CVAppCommon/` | App facade: DB root, views, console, selection |
| `ecvDisplayTools` / `ecvGenericGLDisplay` | `libs/VtkEngine/` | Opacity, light intensity, redraw, multi-view refresh |
| `ccStdPluginInterface` | `libs/CVPluginAPI/` | Standard plugin base; `getActions()`, selection callbacks |
| `AddPlugin()` | `plugins/cmake/Plugins.cmake` | Register plugin target (Standard / IO / GL) |
| `AutomaticReconstructionController` | `app/reconstruction/` | GUI wrapper for automatic SfM/dense pipeline |
| `DA3DepthController` | `libs/Reconstruction/src/controllers/` | DA3 sparse/dense integration with reconstruction |
| `aicore_depth_*` | `core/AICore/include/aicore/depth_capi.h` | DA3 C API |
| `aicore_gaussian_*` | `core/AICore/include/aicore/gaussian_capi.h` | FreeSplatter C API |
| `JsonRPCPlugin` | `plugins/core/Standard/qJSonRPCPlugin/` | WebSocket RPC for agents (port 6001) |
| `ecvPropertiesTreeDelegate` | `app/db_tree/` | DB property panel (opacity, light, recursive group apply) |

Plugin entry: each plugin implements `QObject` + `ccStdPluginInterface`, ships `info.json` + `.qrc`.

## Build Instructions

**Canonical references** (full flag lists, wheel builds, troubleshooting):

| Platform | Guide | Automation |
|----------|-------|------------|
| Linux (Ubuntu) | [compiling-cloudviewer-linux.md](docs/guides/compiling_doc/compiling-cloudviewer-linux.md) | [docker/build-release.sh](docker/build-release.sh), [docker/build-release-conda.sh](docker/build-release-conda.sh) |
| macOS | [compiling-cloudviewer-macos.md](docs/guides/compiling_doc/compiling-cloudviewer-macos.md) | [scripts/build_macos.sh](scripts/build_macos.sh) |
| Windows | [compiling-cloudviewer-windows.md](docs/guides/compiling_doc/compiling-cloudviewer-windows.md) | [scripts/build_win.py](scripts/build_win.py) |

CMake option reference: **[BUILD.md](BUILD.md)**.

### Python environment: Conda vs pyenv

| Platform | Recommended path | Key CMake flags |
|----------|------------------|-----------------|
| **Linux** | **Option A — pyenv + apt** (CI, clean env) | `-DBUILD_WITH_CONDA=OFF`, explicit `-DPython3_EXECUTABLE` / `-DPython3_LIBRARY`, `-DCMAKE_PREFIX_PATH=<Qt>` |
| **Linux** | **Option B — Conda** (reproducible Qt/VTK/CGAL) | `-DBUILD_WITH_CONDA=ON`, `-DCONDA_PREFIX=$CONDA_PREFIX`, `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib` |
| **macOS** | **Conda only** (see guide) | Same as Linux Option B; env from `.ci/conda_macos_cloudViewer.yml` |
| **Windows** | **Conda only** (see guide) | Same as Linux Option B; run `scripts/setup_conda_env.ps1`; env from `.ci/conda_windows_cloudViewer.yml` |

Linux Option A also runs `utils/install_deps_ubuntu.sh assume-yes` and sets up **pyenv** Python 3.10–3.13 before configure. Conda paths on all platforms: create env → `conda activate cloudViewer` → export `PKG_CONFIG_PATH` / `LD_LIBRARY_PATH` (Linux) or `PATH` (macOS) as in the platform guide.

> **Qt note:** Qt 6 only on Ubuntu 24.04+; on 20.04/22.04 use `-DUSE_QT6=OFF`. **macOS:** `PLUGIN_STANDARD_QSIBR=OFF` in CI (OpenGL/Metal limits).

### Parallel jobs and memory

Full builds (many plugins + OpenCV + reconstruction + AICore) can **OOM** on machines with ≤16 GB RAM if you use all cores.

```bash
# Linux — default (enough RAM, e.g. 32GB+)
BUILD_JOBS=$(nproc)

# Linux / macOS — limited RAM: cap jobs (CI uses 4; docs suggest -j4)
BUILD_JOBS=4
# or: BUILD_JOBS=$(( $(nproc) / 2 ))
```

| Platform | Build command |
|----------|---------------|
| Linux (Make) | `make -j"${BUILD_JOBS}"` after `cmake ..` in `build_app/` |
| Linux/macOS (Ninja) | `cmake --build build_app -j "${BUILD_JOBS}"` |
| macOS (Make) | `make -j"$(sysctl -n hw.logicalcpu)"` — or set `BUILD_JOBS` lower if OOM |
| Windows | `$env:NPROC = 4` then `cmake --build . --config Release --parallel $env:NPROC` |

If linking fails with “Killed” or “c++: fatal error: Killed”, reduce `BUILD_JOBS`, add swap, or disable heavy targets (`-DBUILD_CUDA_MODULE=OFF`, fewer `PLUGIN_*`).

### Linux — Option A (pyenv + system packages)

```bash
utils/install_deps_ubuntu.sh assume-yes
# pyenv: install Python 3.12, then resolve paths (see linux guide § A2–A3)

mkdir -p build_app && cd build_app
cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_CONDA=OFF \
    -DCMAKE_PREFIX_PATH="${QT_DIR}" \
    -DPython3_EXECUTABLE="${PYTHON_EXE}" \
    -DPython3_ROOT_DIR="${PYTHON_ROOT}" \
    -DPython3_LIBRARY="${PYTHON_LIB}" \
    -DBUILD_OPENCV=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DUSE_VTK_BACKEND=ON \
    -DUSE_QT6=OFF \
    -DAICore_ENABLED=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QFREESPLATTER=ON \
    ..
make -j"${BUILD_JOBS:-$(nproc)}"
```

### Linux — Option B (Conda)

```bash
conda env create -f .ci/conda_cloudViewer.yml   # adjust Python version per guide
conda activate cloudViewer
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

mkdir -p build_app && cd build_app
cmake \
    -DDEVELOPER_BUILD=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_WITH_CONDA=ON \
    -DCONDA_PREFIX=$CONDA_PREFIX \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
    -DBUILD_OPENCV=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DAICore_ENABLED=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QFREESPLATTER=ON \
    ..
make -j"${BUILD_JOBS:-$(nproc)}"
```

### macOS / Windows

Follow the platform guide; do **not** copy the Linux pyenv recipe on macOS/Windows.

```bash
# macOS (after conda activate)
./scripts/build_macos.sh 2>&1 | tee build.log
```

```powershell
# Windows (PowerShell, after conda activate)
python .\scripts\build_win.py
# or manual: cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_WITH_CONDA=ON ...
#            cmake --build . --config Release --parallel $env:NPROC
```

### Feature toggles (add to any configure line)

```bash
# Manual calibration
-DPLUGIN_STANDARD_QMANUAL_CALIB=ON

# Agent JSON-RPC
-DPLUGIN_STANDARD_QJSONRPC=ON

# qSIBR viewer (Linux/Windows; OFF on macOS)
-DPLUGIN_STANDARD_QSIBR=ON

# GPU (Linux/Windows; not macOS)
-DBUILD_CUDA_MODULE=ON -DBUILD_COMMON_CUDA_ARCHS=ON
```

| Option | Role |
|--------|------|
| `BUILD_WITH_CONDA` | ON = Conda-managed deps; OFF = system apt + pyenv (Linux only) |
| `AICore_ENABLED` | Build `libAICore.so`; auto-enables `GGML_ENABLED` |
| `BUILD_RECONSTRUCTION` | COLMAP + automatic reconstruction UI |
| `BUILD_CUDA_MODULE` | CloudViewer core CUDA (algorithms, pybind, qSIBR); independent of `AICore_USE_CUDA` |
| `PLUGIN_STANDARD_*` | Per-plugin toggles (see `BUILD.md`) |
| `MCALIB_BUILD_TESTS` / `MCALIB_BUILD_TOOLS` | qManualCalib tests and CLI tools |

**Outputs:** Linux/macOS → `build_app/bin/ACloudViewer`, `build_app/bin/libAICore.so`, `build_app/bin/plugins/libQ*_PLUGIN.so`; Windows → `build_app/bin/Release/ACloudViewer.exe` (plus plugins under `Release/`).

## Testing

```bash
# C++ unit tests (when BUILD_UNIT_TESTS=ON)
cd build_app && ctest --output-on-failure

# qDA3 / AICore tests (need GGUF assets; missing → exit 77 skip)
cmake -DAICore_ENABLED=ON -DAICore_BUILD_TESTS=ON ..
cmake --build build_app --target test_capi -j "${BUILD_JOBS:-4}"

# qManualCalib bag reader
cmake -DPLUGIN_STANDARD_QMANUAL_CALIB=ON -DMCALIB_BUILD_TESTS=ON ..
cmake --build build_app --target test_bag_reader -j
./build_app/bin/plugins/test_bag_reader

# Agent integration (Python, separate harness repo)
pytest cli_anything/acloudviewer/tests/ -v
```

Test data: `examples/test_data/` (CMake download list); qManualCalib ships `plugins/core/Standard/qManualCalib/tests/data/`.

## Documentation

| Audience | Location |
|----------|----------|
| Build / CMake | `BUILD.md`, `docs/guides/compiling_doc/` |
| Plugin catalog | `plugins/README.md` |
| AI user guides | `docs/guides/plugins/` (qDA3, qFreeSplatter, qManualCalib) |
| Per-plugin dev docs | `plugins/core/<Category>/<Plugin>/README.md` |
| Model / sample data cards | `plugins/core/Standard/q*/models/MODEL_CARD.md`, `qManualCalib/tests/data/DATA_CARD.md` |
| Sphinx API | `docs/source/` (plugin READMEs synced at doc-build via `docs/source/conf.py`) |
| Agents | `agent-integration/README.md` |

**When editing docs:** prefer **incremental** changes — add rows/sections for new plugins; do not rewrite existing paragraphs, retitle files, or merge unrelated links into one line (reduces merge conflicts).

## Code Style & Conventions

### Naming (CloudCompare heritage + newer code)

| Kind | Convention | Examples |
|------|------------|----------|
| Core entity classes | `cc` + PascalCase | `ccPointCloud`, `ccHObject`, `ccMesh` |
| App / engine classes | `ecv` prefix | `ecvMainAppInterface`, `ecvDisplayTools` |
| Plugins | `q` + PascalCase folder | `qDA3`, `qFreeSplatter`, `qManualCalib` |
| Plugin CMake target | `QDA3_PLUGIN`, `QFREESPLATTER_PLUGIN` | Uppercase + `_PLUGIN` |
| Files | Often camelCase with prefix | `ecvPointCloud.cpp`, `DA3Dialog.cpp` |
| CMake options | `PLUGIN_STANDARD_QDA3`, `AICore_ENABLED` | UPPER_SNAKE |
| DB export naming | `ecvPluginDbNaming` | Prefixed entity names for plugin outputs |

New AICore / reconstruction code may use `snake_case` for functions and `PascalCase` for types; match the **surrounding file**.

### UI performance (VTK property panel)

- **Slider drag:** lightweight VTK preview + debounced `renderScene()`; avoid full DB/representation rebuild on every tick.
- **Slider release / spinbox commit:** full sync (`ensureRepresentation`, `changeEntityProperties`, refresh).
- **Folder recursion:** use `obj->isGroup()` (`HIERARCHY_OBJECT`), not merely `getChildrenNumber() > 0`.

### Plugins

- Standard: `plugins/core/Standard/<Name>/` — `CMakeLists.txt`, `info.json`, `.qrc`, `README.md`
- I/O: `plugins/core/IO/<Name>/`
- Register with `AddPlugin(NAME ...)`; link `CVCoreLib`, `CVPluginAPI`, `CVPluginStub`
- Scoped Cursor rules: `.cursor/rules/acloudviewer-plugin-dev.mdc`

### Agent integration

- RPC methods: `category.action` in `JsonRPCPlugin::execute()`; update `rpcMethodsList()`
- Scoped rules: `.cursor/rules/acloudviewer-agent-dev.mdc`

### Formatting

- C++: clang-format (project history references clang-format-10); match surrounding style in each module
- Python: yapf (see CHANGELOG); agent harness follows Click + pytest patterns

## External Dependencies (summary)

| Library | Role |
|---------|------|
| Qt 5.12+ / 6.2+ | GUI, plugins, concurrent |
| Eigen3 | Linear algebra (core, reconstruction, AICore) |
| VTK | Rendering (`libs/VtkEngine/`) |
| OpenCV | Optional; required for qManualCalib, some reconstruction / image paths |
| ggml | ML inference backend in AICore |
| CUDA / Vulkan / Metal / OpenCL | Optional GPU (core, ggml, BEV in qManualCalib, qSIBR) |
| FFmpeg | Optional H.264/HEVC in qManualCalib bag decode |
| COLMAP stack | Bundled under `libs/Reconstruction/` when `BUILD_RECONSTRUCTION=ON` |

Large downloads: [cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads) (GGUF releases `DA3`, `3dgs`; test assets in repo or `examples/test_data/download_file_list.json`).

## Notable Plugins (quick index)

| Plugin | CMake | Notes |
|--------|-------|-------|
| qDA3 | `PLUGIN_STANDARD_QDA3` | Depth/pose/COLMAP; needs `AICore_ENABLED` |
| qFreeSplatter | `PLUGIN_STANDARD_QFREESPLATTER` | 3D Gaussian splats; optional qSIBR viewer |
| qManualCalib | `PLUGIN_STANDARD_QMANUAL_CALIB` | Sensor/AVM calibration; sample data in-tree |
| qSIBR | `PLUGIN_STANDARD_QSIBR` | Gaussian / ULR viewers (CUDA, Linux/Win) |
| qJSonRPCPlugin | `PLUGIN_STANDARD_QJSONRPC` | Agent WebSocket API |

Full table: [plugins/README.md](plugins/README.md) and [BUILD.md](BUILD.md).

## CI & Release

- Workflows: `.github/workflows/ubuntu.yml`, `macos.yml`, `windows.yml`, `documentation.yml`, `agent-integration.yml`, `codeql.yml`
- Local CI helpers: `util/ci_utils.sh` (Linux), `util/ci_utils.ps1` (Windows)
- Version: `libs/cloudViewer/version.txt`; changelog: `CHANGELOG.md`


<claude-mem-context>
# Memory Context

# [ACloudViewer] recent context, 2026-07-23 7:27pm GMT+8

No previous sessions found.
</claude-mem-context>