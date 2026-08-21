# AGENTS.md — ACloudViewer Guide

**Version:** 3.9.5 · **Language:** C++17 · **GUI:** Qt 5/6

Reference layout: Use this file when you need a **full-repo map** (build, modules, conventions). For scoped rules, also read `.agents/rules/*.mdc`.

## For AI Agents — Start Here

| Your task | Read first | Then |
|-----------|------------|------|
| Control / automate ACloudViewer | This file § Agent integration | `agent-integration/README.md`, `agent-integration/docs/CLI-QUICK-REFERENCE.md` |
| Build or fix compile/link errors | This file § Build Instructions | Platform guide in `docs/guides/compiling_doc/`, `BUILD.md` |
| Develop a plugin | `.agents/rules/acloudviewer-plugin-dev.mdc` | `plugins/core/<Category>/<Plugin>/README.md` |
| Add JSON-RPC / MCP / CLI command | `.agents/rules/acloudviewer-agent-dev.mdc` | `agent-integration/docs/JSON-RPC-API.md` |
| Modify ggml / AICore / GPU backend | `.agents/rules/acloudviewer-ggml-aicore.mdc` | This file § ggml 源码修改规则, `3rdparty/ggml/patches/` |
| Debug CI failure | `.agents/rules/acloudviewer-ci-debugging.mdc` | `.github/workflows/`, `util/ci_utils.sh` |
| Understand a module | This file § Module Layers + Key Classes | Per-plugin README, Sphinx `docs/source/` |

**Hard rules for AI operators:**

1. **Never guess binary CLI flags** — use `cli-anything-acloudviewer` (headless) or JSON-RPC (GUI).
2. **Windows file ops** — prefer `--mode headless` to avoid RPC hang when port 6001 is stale.
3. **ggml 源码修改** — **禁止**手动改 ggml 源码（含 `build*/ggml/`、vendor tarball、任何临时提取目录）。所有改动**必须**以 unified diff patch 提交到 `3rdparty/ggml/patches/`，由 CMake ExternalProject 在构建时通过 `apply_ggml_patches.py`（`git apply`）自动应用；见 § ggml 源码修改规则。
4. **Doc edits** — incremental additions only; do not rewrite unrelated sections (reduces merge conflicts).

## Project Overview

ACloudViewer is an open-source **3D point cloud and mesh processing** application and library, descended from CloudCompare with integrations for Open3D, ParaView-style visualization, **COLMAP** reconstruction, VTK rendering, Python bindings, and optional **AI inference** (Depth Anything V3, FreeSplatter via ggml). Primary language: **C++17**. GUI: **Qt 5/6**. Optional **CUDA**, **Vulkan** (Linux/Windows), **Metal** (macOS), and **Python 3.10+**.

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

> **AI 操作 ACloudViewer（重要）**：所有面向 AI / 自动化脚本的 CLI 交互，**必须**通过 `agent-integration/` 提供的 `cli-anything-acloudviewer` 工具链（headless 直接调用二进制，GUI 走 JSON-RPC）。**不要**直接猜二进制参数。先读 `agent-integration/README.md` 与 `agent-integration/docs/CLI-QUICK-REFERENCE.md`。安装：`pip install git+https://github.com/Asher-1/CLI-Anything.git#subdirectory=acloudviewer/agent-harness`（本机已装 v3.1.0）。

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
| `.agents/` | Cursor rules (`.mdc`), skills, MCP config (`mcp.json`) |
| `.ci/` | Conda environment YAMLs per platform / Qt version |

### Cursor Agent Configuration (`.agents/`)

| Path | Purpose |
|------|---------|
| `.agents/rules/acloudviewer-plugin-dev.mdc` | Plugin architecture, `AddPlugin()`, entity types |
| `.agents/rules/acloudviewer-agent-dev.mdc` | JSON-RPC, MCP, CLI harness development |
| `.agents/rules/acloudviewer-ggml-aicore.mdc` | ggml ExternalProject, patches, GPU backends |
| `.agents/rules/acloudviewer-ci-debugging.mdc` | CI matrix, Docker layers, platform GPU policy |
| `.agents/rules/pua.mdc` | Escalation/debugging methodology (after repeated failures) |
| `.agents/mcp.json` | Pre-configured MCP servers: `acloudviewer`, `-headless`, `-gui` |
| `.agents/skills/` | Repo-local skills (`codebase-summarizer`, `first-principles`, …) |

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
| `ecvViewManager` | `libs/VtkEngine/` | Multi-window view registry (v3.9.5+); per-view camera, VTK widget, display state |
| `JsonRPCResult` | `plugins/core/Standard/qJSonRPCPlugin/` | RPC success/error envelope for all agent methods |

Plugin entry: each plugin implements `QObject` + `ccStdPluginInterface`, ships `info.json` + `.qrc`.

### Multi-View Architecture (v3.9.5+)

ACloudViewer supports multiple independent 3D/chart/ortho/comparative views (ParaView-style). Key implications for agents and plugin authors:

- `ecvDisplayTools` is **per-view**, not a global singleton — pass `ecvViewContext&` where applicable.
- Camera links, ortho slices, and chart views each have dedicated view types; redraw targets a specific view.
- JSON-RPC view methods operate on the active view unless an explicit view ID is provided.
- When debugging rendering issues, check `ecvViewManager` and per-view VTK widget state before assuming a global display bug.

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
| **Linux** | **Option B — Conda** (reproducible Qt/VTK/CGAL) | `-DBUILD_WITH_CONDA=ON`, `-DCONDA_PREFIX=$CONDA_PREFIX`, `-DCMAKE_PREFIX_PATH=$CONDA_PREFIX` |
| **macOS** | **Conda only** (see guide) | Same as Linux Option B; env from `.ci/conda_macos_cloudViewer.yml` |
| **Windows** | **Conda only** (see guide) | Same as Linux Option B; run `scripts/setup_conda_env.ps1`; env from `.ci/conda_windows_cloudViewer.yml` |

Linux Option A also runs `util/install_deps_ubuntu.sh assume-yes` and sets up **pyenv** Python 3.10–3.13 before configure. Conda paths on all platforms: create env → `conda activate cloudViewer` → export `PKG_CONFIG_PATH` / `LD_LIBRARY_PATH` (Linux) or `PATH` (macOS) as in the platform guide.

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
util/install_deps_ubuntu.sh assume-yes
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
    -DPLUGIN_STANDARD_QDEEPLSD=ON \
    -DPLUGIN_STANDARD_QFACEDETECT=ON \
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
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX \
    -DBUILD_OPENCV=ON \
    -DBUILD_RECONSTRUCTION=ON \
    -DAICore_ENABLED=ON \
    -DPLUGIN_STANDARD_QDA3=ON \
    -DPLUGIN_STANDARD_QDEEPLSD=ON \
    -DPLUGIN_STANDARD_QFACEDETECT=ON \
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

### Incremental Build Tips

```bash
# Reconfigure after CMake option change
cd build_app && cmake ..

# Rebuild single target
cmake --build build_app --target ACloudViewer -j "${BUILD_JOBS:-4}"
cmake --build build_app --target QDA3_PLUGIN -j4

# After ggml.cmake config change — delete ExternalProject stamp first
rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{install,done}
cmake --build build_app --target ext_ggml -j4
```

### Python Package (optional)

Requires `-DBUILD_PYTHON_MODULE=ON` (default ON). From `build_app/`:

```bash
make python-package          # build pybind module
make pip-package             # build wheel
make install-pip-package     # pip install the wheel
pip uninstall cloudViewer    # remove
```

Wheel runtime checks: `docker/test_wheel_runtime.sh`, `check_aicore_runtime.py` (Docker CI is CPU-only; no Vulkan device required).

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
- Scoped Cursor rules: `.agents/rules/acloudviewer-plugin-dev.mdc`

### Agent integration

- RPC methods: `category.action` in `JsonRPCPlugin::execute()`; update `rpcMethodsList()`
- Scoped rules: `.agents/rules/acloudviewer-agent-dev.mdc`
- **三套接口**（详见 `agent-integration/README.md`）：
  - **JSON-RPC**（WebSocket 6001，GUI 实时控制）— 插件 `qJSonRPCPlugin`，`PLUGIN_STANDARD_QJSONRPC=ON`
  - **MCP Server**（stdio，供 OpenClaw/Cursor/Claude Code）— `cli-anything-acloudviewer-mcp`；Cursor 一键配置见 `.agents/mcp.json`
  - **CLI Harness**（Click，headless 直接调二进制 / GUI 走 RPC）— `cli-anything-acloudviewer`
- **CLI 常用操作**（headless 无需 GUI）：
  - `cli-anything-acloudviewer info` / `formats` — 环境与格式
  - `cli-anything-acloudviewer --mode headless convert in.ply out.obj` — 格式转换（Windows 推荐加 `--mode headless`）
  - `cli-anything-acloudviewer process <op> in.ply -o out.ply` — 55+ 处理算子（subsample/normals/crop/icp/csf/ransac/m3c2/canupo/poisson/cork 等）
  - `cli-anything-acloudviewer reconstruct auto ./imgs -w ./ws` — COLMAP 重建
  - `cli-anything-acloudviewer view screenshot out.png` — GUI 截图（需 GUI）
  - `cli-anything-acloudviewer --json scene list` — GUI 场景树
- **运行 Python 脚本（qPythonRuntime）**：GUI 模式下在插件面板手动运行；CLI 可用 `ACloudViewer -SILENT -PYTHON_SCRIPT x.py`（headless）。脚本示例见 `plugins/core/Standard/qPythonRuntime/script_examples/`

**Agent integration docs:**

| Doc | Content |
|-----|---------|
| `agent-integration/docs/CLI-QUICK-REFERENCE.md` | Full CLI command reference (55+ process ops) |
| `agent-integration/docs/COMMAND-MAPPING.md` | CLI ↔ JSON-RPC ↔ binary flag mapping |
| `agent-integration/docs/JSON-RPC-API.md` | WebSocket RPC method catalog |
| `agent-integration/docs/SIBR-VIEWER-CLI.md` | SIBR Gaussian/ULR viewer commands |
| `agent-integration/docs/TROUBLESHOOTING.md` | RPC hang, headless vs GUI, path issues |
| `agent-integration/docs/TESTING.md` | Harness pytest, E2E with `ACLOUDVIEWER_E2E_GUI=1` |

**Common agent mistakes:**

| Mistake | Fix |
|---------|-----|
| Guessing `ACloudViewer` command-line flags | Use `cli-anything-acloudviewer` or read `COMMAND-MAPPING.md` |
| Windows `auto` mode hangs on convert | Add `--mode headless` |
| RPC returns empty / connection refused | Enable qJSonRPCPlugin, toggle server on port 6001 |
| Headless crop with wrong bounds | Use six separate `--min-x` … `--max-z` flags (not colon-separated) |
| Expecting GPU in Docker CI | CPU-only is valid; do not require `--expect-device vulkan` |

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
| CUDA / Vulkan / Metal | Optional GPU (core, ggml, BEV in qManualCalib, qSIBR); Vulkan Linux/Windows only, Metal macOS only |
| FFmpeg | Optional H.264/HEVC in qManualCalib bag decode |
| COLMAP stack | Bundled under `libs/Reconstruction/` when `BUILD_RECONSTRUCTION=ON` |

Large downloads: [cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads) (GGUF releases `DA3`, `3dgs`; test assets in repo or `examples/test_data/download_file_list.json`).

### Platform ggml Backend Support

| Platform | Default GPU | Auto Device Order | Vulkan | Metal | CUDA | Notes |
|----------|------------|-------------------|--------|-------|------|-------|
| **macOS** | Metal | Metal → CPU | OFF (unsupported) | ON | OFF | MoltenVK translation limitations prevent Vulkan use |
| **Linux** | Vulkan | Vulkan → CPU | ON | OFF | Optional | CUDA priority when enabled: CUDA → Vulkan → CPU |
| **Windows** | Vulkan | Vulkan → CPU | ON | OFF | Optional | Same priority as Linux |

> **macOS Vulkan defect:** MoltenVK SPIR-V → MSL translation fails for complex ggml compute shaders (conv_transpose, quantized matmul). Metal is both native and faster. Vulkan support was removed from macOS builds in v3.9.5.

### ggml 源码修改规则（强制）

> **所有涉及 ggml 源码的修改，都必须通过 CMake 构建链自动 apply patch 实现，禁止手动改 ggml 源码。**

ggml 在本仓库中是 **ExternalProject**（`3rdparty/ggml/ggml.cmake`）：每次 configure/干净构建都会从 tarball **重新解压**一份全新源码，随后在 `PATCH_COMMAND` 阶段由 `3rdparty/ggml/patches/apply_ggml_patches.py` 按 `manifest.yaml` 顺序执行 `git apply`。

#### 禁止的做法

| 禁止 | 后果 |
|------|------|
| 直接编辑 `build/ggml/`、`build_app/ggml/` 下的 `.c/.cpp/.h/.metal` 等 | 本地可能暂时能编过，但**无法提交**、CI/他人/干净构建**全部丢失** |
| 把 ggml 改动只留在工作区而未生成 patch | PR 无 diff、修复**无法合入**、问题会在下次构建复现 |
| 绕过 `manifest.yaml` 手工 `patch -p1` 后不再入库 | 不可复现，团队与 CI 行为不一致 |
| 在 ggml 上游目录直接改 vendor 副本而不走 patch 流程 | 同上；vendor 树会被 ExternalProject 覆盖 |

**允许的唯一路径：** patch 文件入库 → `manifest.yaml` 注册 → CMake 构建时自动 apply。

#### 正确流程

```
1. 在 build_app/ggml/... 的解压副本中**临时**修改并验证（仅作试验场，不要提交这里的文件）
2. 生成 unified diff：
     diff -ruN orig/ modified/ > 3rdparty/ggml/patches/<subdir>/0001-描述.patch
   或对比 git 状态生成 patch
3. 将 *.patch 放入 3rdparty/ggml/patches/<子目录>/
4. 在 3rdparty/ggml/patches/manifest.yaml 中注册（顺序重要，见已有条目）
5. 清理 ExternalProject stamp 后重建，确认 apply_ggml_patches.py 成功：
     rm -f build_app/ggml/src/ext_ggml-stamp/ext_ggml-{install,done}
     cmake --build build_app --target ext_ggml -j4
6. 仅提交 patch + manifest.yaml（及必要的 ggml.cmake / AICore 胶水代码），**不要**提交 build*/ggml/ 下的源码
```

目录结构：

```
3rdparty/ggml/patches/
├── manifest.yaml              # 所有 patch 的有序清单（唯一权威来源）
├── apply_ggml_patches.py      # 构建时 CMake 调用；内部用 git apply --directory
├── aliked_merged/0001-*.patch
├── cpu_all_variants/0001-*.patch
├── metal_merged/0001-*.patch
└── msvc_vulkan/0001-*.patch
```

#### 为何必须如此

- ExternalProject 每次从 tarball 解压 → **构建目录里的 ggml 源码不是持久状态**。
- `apply_ggml_patches.py` 会先在前向 replay 验证整条 patch 链，再 `git apply` → 保证跨平台、可复现、可审查。
- 历史上 in-place Python 改源码的脚本已**全部迁移**为 unified diff patch；新工作必须继续走 patch 流程。

详细后端变量与调试：`.agents/rules/acloudviewer-ggml-aicore.mdc`（`AICore_USE_CUDA`、`AICore_BUNDLE_CUDA_RUNTIME` 等）。

## Notable Plugins (quick index)

| Plugin | CMake | Notes |
|--------|-------|-------|
| qDA3 | `PLUGIN_STANDARD_QDA3` | Depth/pose/COLMAP; needs `AICore_ENABLED` + `BUILD_RECONSTRUCTION` for auto recon |
| qDeepLSD | `PLUGIN_STANDARD_QDEEPLSD` | Line segment detection (GGUF); needs `AICore_ENABLED` |
| qFaceDetect | `PLUGIN_STANDARD_QFACEDETECT` | Face detection/embedding (GGUF); needs `AICore_ENABLED` |
| qLightGlue | `PLUGIN_STANDARD_QLIGHTGLUE` | Feature matching (GGUF); needs `AICore_ENABLED` |
| qFreeSplatter | `PLUGIN_STANDARD_QFREESPLATTER` | 3D Gaussian splats; optional qSIBR viewer |
| qRFDetr | `PLUGIN_STANDARD_QRFDETR` | RF-DETR detection/segmentation (GGUF); needs `AICore_ENABLED` |
| qRMBG | `PLUGIN_STANDARD_QRMBG` | RMBG-2.0 background removal (GGUF); needs `AICore_ENABLED` |
| qYOLO | `PLUGIN_STANDARD_QYOLO` | YOLO detection + segmentation + metric depth (GGUF); needs `AICore_ENABLED` |
| qManualCalib | `PLUGIN_STANDARD_QMANUAL_CALIB` | Sensor/AVM calibration; sample data in-tree |
| qPythonRuntime | `PLUGIN_PYTHON` | In-app Python scripting; headless via `-PYTHON_SCRIPT` |
| qSIBR | `PLUGIN_STANDARD_QSIBR` | Gaussian / ULR viewers (CUDA, Linux/Win) |
| qJSonRPCPlugin | `PLUGIN_STANDARD_QJSONRPC` | Agent WebSocket API |

All AICore-backed plugins share one `libAICore.so` with a single ggml copy. Model cards: `plugins/core/Standard/q*/models/MODEL_CARD.md`. Downloads: [cloudViewer_downloads](https://github.com/Asher-1/cloudViewer_downloads) (GGUF tags `DA3`, `3dgs`, …).

Full table: [plugins/README.md](plugins/README.md) and [BUILD.md](BUILD.md).

## CI & Release

- Workflows: `.github/workflows/ubuntu.yml`, `macos.yml`, `windows.yml`, `documentation.yml`, `agent-integration.yml`, `codeql.yml`
- Local CI helpers: `util/ci_utils.sh` (Linux), `util/ci_utils.ps1` (Windows)
- Docker CI: `docker/Dockerfile.ci`, `docker/docker_test.sh`, `docker/build-release.sh`
- Version: `libs/cloudViewer/version.txt`; changelog: `CHANGELOG.md`

### CI Matrix (quick reference)

| Platform | Workflow | GPU backend | Notable constraints |
|----------|----------|-------------|---------------------|
| Ubuntu 20.04/22.04/24.04 | `ubuntu.yml` | Vulkan (CPU-only in Docker) | focal: shaderc from source; 24.04: Qt6 option |
| macOS 13+ | `macos.yml` | Metal only | `PLUGIN_STANDARD_QSIBR=OFF`, `GGML_OPENMP=OFF` |
| Windows 10/11 | `windows.yml` | Vulkan | Conda-only; path-space quoting in patch scripts |
| CUDA variants | `ubuntu.yml` | CUDA → Vulkan → CPU | `AICore_USE_CUDA=ON` when `BUILD_CUDA_MODULE=ON` |

Debug workflow: read CI log **bottom-up** for the first error; distinguish Docker build layer vs test phase. See `.agents/rules/acloudviewer-ci-debugging.mdc`.

## Troubleshooting (agent quick ref)

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| Link killed / OOM | Too many parallel jobs | Set `BUILD_JOBS=4`, disable heavy plugins |
| `glslc is missing` (focal) | Vulkan install script stdout pollution | See CI rules; build output must go to stderr |
| macOS crash on AICore load | Duplicate OpenMP | Verify `GGML_OPENMP=OFF`, `otool -L libggml-cpu.so \| grep omp` empty |
| GPU inference fails on target machine | Missing CUDA runtime libs | `-DAICore_BUNDLE_CUDA_RUNTIME=ON`; launcher adds `lib/cuda-runtime/` to path |
| RPC / CLI hang (Windows) | Stale port 6001 | `--mode headless` |
| AICore test skip (exit 77) | Missing GGUF model assets | Download from cloudViewer_downloads or set skip |
| Plugin not in menu | CMake option OFF or build target missing | Reconfigure with `-DPLUGIN_STANDARD_Q…=ON`, rebuild plugin target |
| ggml fix works locally but not in CI/PR | 手动改了 `build*/ggml/` 未生成 patch | 按 § ggml 源码修改规则 生成 patch 并注册 `manifest.yaml` |


<claude-mem-context>
# Memory Context

# [ACloudViewer] recent context, 2026-08-20 3:59pm GMT+8

No previous sessions found.
</claude-mem-context>