# Building ACloudViewer from Source

Complete guide for building ACloudViewer from source code on Windows, Linux, and macOS.

---

## 🚀 Quick Start

**Choose your platform and follow the detailed build guide:**

| Platform | Complete Build Guide | Quick Build Script |
|----------|---------------------|-------------------|
| 🪟 **Windows** | **[Windows Build Guide →](docs/guides/compiling_doc/compiling-cloudviewer-windows.md)** | `python scripts/build_win.py` |
| 🐧 **Linux** | **[Linux Build Guide →](docs/guides/compiling_doc/compiling-cloudviewer-linux.md)** | `./docker/build-release.sh` |
| 🍎 **macOS** | **[macOS Build Guide →](docs/guides/compiling_doc/compiling-cloudviewer-macos.md)** | `./scripts/build_macos.sh` |

> 💡 **First time building?** The platform-specific guides include detailed setup instructions, dependency installation, and troubleshooting tips.

**Additional Resources:**
- 📚 **Comprehensive Reference**: [QUICKSTART.md](docs/guides/QUICKSTART.md) - Build options, environment setup, and advanced configuration
- 🤖 **Agent Integration**: [agent-integration/README.md](agent-integration/README.md) - For AI agent / automation development

---

## 📋 System Requirements

### Supported Platforms

| Platform | Version | Compiler | Architecture |
|----------|---------|----------|--------------|
| **Windows** | 10/11 (64-bit) | Visual Studio 2022 | x64 |
| **Ubuntu** | 20.04, 22.04, 24.04 | GCC 9+, Clang 10+ | x64 |
| **macOS** | 10.14+ | XCode 8.0+ | ARM64 |

### Core Build Dependencies

**Required:**
- **CMake** 3.19 or newer
- **Python** 3.10+ (for build scripts)
- **Qt** 5.12+ or Qt 6.2+
- **C++17** compliant compiler

**Recommended:**
- **Conda/Miniconda** (for dependency management)
- **CUDA** 12.0+ (for GPU acceleration)

---

## 🏗️ Build Targets

ACloudViewer supports multiple build configurations:

### Application Builds

```bash
# GUI Application (ACloudViewer)
cmake --build . --config Release --target ACloudViewer

# Viewer Application (CloudViewer)
cmake --build . --config Release --target CloudViewer

# Reconstruction Tools (COLMAP integration)
cmake --build . --config Release --target COLMAP
```

### Python Package

```bash
# Build Python package
make python-package

# Build and install pip wheel
make pip-package
make install-pip-package

# Uninstall
pip uninstall cloudViewer
```

### Full Install

```bash
# Install all components (binaries, plugins, resources)
cmake --build . --config Release --target install
```

---

## 🔧 CMake Configuration Options

### Core Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_SHARED_LIBS` | OFF | Build shared libraries instead of static |
| `BUILD_EXAMPLES` | ON | Build example programs |
| `BUILD_UNIT_TESTS` | OFF | Build unit tests |
| `BUILD_BENCHMARKS` | OFF | Build micro benchmarks |
| `BUILD_PYTHON_MODULE` | ON | Build Python bindings (cloudViewer package) |
| `BUILD_GUI` | ON | Build GUI applications (ACloudViewer, CloudViewer) |
| `BUILD_RECONSTRUCTION` | OFF | Build COLMAP 3D reconstruction integration |
| `BUILD_DOCUMENTATION` | OFF | Build Doxygen + Sphinx documentation |

### CUDA & GPU Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_CUDA_MODULE` | OFF | Enable CUDA GPU acceleration |
| `BUILD_WITH_CUDA_STATIC` | ON | Use static CUDA libraries |
| `BUILD_COMMON_CUDA_ARCHS` | OFF | Build for common GPU architectures (for release) |
| `BUILD_CACHED_CUDA_MANAGER` | ON | Enable cached CUDA memory manager |

### Rendering & Visualization

| Option | Default | Description |
|--------|---------|-------------|
| `USE_VTK_BACKEND` | ON | Use VTK as rendering backend |
| `ENABLE_HEADLESS_RENDERING` | OFF | Use OSMesa for headless rendering (no GPU) |
| `USE_QT6` | OFF | Use Qt6 instead of Qt5 (requires Qt6 6.2+) |

### Performance & Optimization

| Option | Default | Description |
|--------|---------|-------------|
| `WITH_OPENMP` | ON | Enable OpenMP multi-threading |
| `WITH_IPP` | ON | Use Intel Integrated Performance Primitives |
| `USE_SIMD` | OFF | Enable SIMD optimizations |
| `STATIC_WINDOWS_RUNTIME` | OFF | Use static (MT/MTd) Windows runtime |

### Machine Learning Integration

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_TENSORFLOW_OPS` | OFF | Build TensorFlow operators |
| `BUILD_PYTORCH_OPS` | OFF | Build PyTorch operators |
| `BUNDLE_CLOUDVIEWER_ML` | OFF | Include CloudViewer-ML repo in Python wheel |
| `AICore_ENABLED` | OFF | Build unified AI core (`libAICore.so`) — DA3 depth/pose + FreeSplatter 3D Gaussians (auto-enables `GGML_ENABLED`) |
| `AICore_BUILD_TESTS` | OFF | Build the lightweight public ABI/runtime contract suite |
| `AICore_BUILD_WHITEBOX_TESTS` | OFF | Also build private implementation tests (slower; duplicates AICore objects) |
| `GGML_ENABLED` | (internal) | Auto-synced from `AICore_ENABLED`; not a separate user switch |
| `AICore_USE_METAL` | Apple: ON, else OFF | Metal backend (Apple only; macOS Auto default) |
| `AICore_USE_VULKAN` | Linux/Win: ON, macOS: OFF | **ON:** build Vulkan backend; configure **fails** if glslc/Vulkan/SPIR-V deps missing |
| `AICore_USE_CUDA` | OFF | Developer CUDA backend (independent of `BUILD_CUDA_MODULE`). When ON, Auto becomes **CUDA → Vulkan → CPU** on Linux/Windows |
| `AICore_USE_SYCL` | OFF | Intel GPU backend; requires oneAPI compiler and a validated runtime bundle |
| `AICore_SYCL_USE_DNN` | ON | oneDNN kernels in SYCL backend (requires `AICore_USE_SYCL=ON`) |
| `AICore_USE_OPENCL` | OFF | Legacy/Adreno developer opt-in; not part of desktop distributions |
| `AICore_OPENCL_TARGET_VERSION` | 200 | OpenCL host API target: 120, 200, or 300 |
| `AICore_BUNDLE_CUDA_RUNTIME` | OFF | **`option()`** in `cmake/AICoreOptions.cmake`: redist CUDA runtime (`lib/cuda-runtime/`) into installer; **requires `AICore_USE_CUDA=ON`**; not in GitHub CI |
| `AICore_CPU_ALL_VARIANTS` | OFF | Build all ggml CPU ISA variants (`libggml-cpu-*.so`; compiler-adaptive; CI release/wheel default ON). Matches [llama.cpp release](https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/release.yml) flags: `-DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CPU_ALL_VARIANTS=ON`. Older GCC (e.g. Ubuntu 20.04) skips BF16/AMX/VNNI variants automatically. |
| `AICore_METAL_ENABLED` | (auto) | Read-only: ON when Metal backend was built |
| `AICore_VULKAN_ENABLED` | (auto) | Read-only: ON when Vulkan backend was built |
| `AICore_CUDA_ENABLED` | (auto) | Read-only: ON when CUDA backend was built |
| `AICore_SYCL_ENABLED` | (auto) | Read-only: ON when SYCL backend was built |
| `AICore_OPENCL_ENABLED` | (auto) | Read-only: ON when OpenCL backend was built |

Compile-time macros (injected by CMake into `libAICore` / tests — **not** user `-D` flags):

| Macro | When set |
|-------|----------|
| `AICORE_BACKEND_DL` | Dynamic ggml backend modules (default packaging) |
| `AICORE_CUDA_STATIC_LINKED` | CUDA statically linked into libAICore (non-DL dev builds) |
| `AICORE_AUTO_INCLUDE_CUDA` | CUDA in Auto fallback (`AICore_CUDA_ENABLED`) |

> **AICore CMake naming:** All switches (`AICore_ENABLED`, tests, backends,
> packaging) are defined in `cmake/AICoreOptions.cmake`. ggml keeps internal `GGML_*`
> names synced automatically; **`-DGGML_*` is ignored** (stale cache entries are
> cleared with a warning). After upgrading from older builds, run
> `cmake -U GGML_USE_METAL -U GGML_USE_VULKAN ...` once or use a fresh build dir.
> When adding a new option, update `cmake/AICoreOptions.cmake`, this table,
> `util/ci_utils.{sh,ps1}`, compile guides, and `core/AICore/README.md`.

### Sensor Support

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_LIBREALSENSE` | OFF | Intel RealSense camera support |
| `BUILD_AZURE_KINECT` | OFF | Azure Kinect sensor support |

### Build Environment

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_WITH_CONDA` | OFF | Build with Conda environment |
| `DEVELOPER_BUILD` | ON | Add +commit_hash to version (for development) |
| `PACKAGE` | OFF | Create installer package after build |
| `GLIBCXX_USE_CXX11_ABI` | ON | Use C++11 ABI (Linux) |

### System Libraries (Advanced)

Use pre-installed system libraries instead of bundled versions:

| Option | Default | Description |
|--------|---------|-------------|
| `USE_SYSTEM_EIGEN3` | OFF | Use system Eigen3 |
| `USE_SYSTEM_OPENCV` | OFF | Use system OpenCV |
| `USE_SYSTEM_PCL` | OFF | Use system PCL |
| `USE_SYSTEM_VTK` | OFF | Use system VTK |
| `USE_SYSTEM_FLANN` | OFF | Use system FLANN |
| `USE_SYSTEM_ASSIMP` | OFF | Use system Assimp |
| `USE_SYSTEM_GLFW` | OFF | Use system GLFW |
| `USE_SYSTEM_GLEW` | OFF | Use system GLEW |
| `USE_SYSTEM_FMT` | OFF | Use system fmt |
| `USE_SYSTEM_PYBIND11` | OFF | Use system pybind11 |
| `USE_SYSTEM_TBB` | OFF | Use system TBB |
| `USE_BLAS` | OFF (ON for ARM) | Use BLAS/LAPACK instead of MKL |

> ⚠️ **Note**: Using system libraries may cause compatibility issues. Only enable if you know what you're doing.

### Plugin Options

Expand the `INSTALL` group in CMake GUI to enable plugins:

#### Standard Plugins

|       Plugin Name       |         CMake Option                     | Default Value | Description
|-------------------------|------------------------------------------|---------------|-------------
| q3DMASC                 | PLUGIN_STANDARD_3DMASC                   | OFF           | Automatic point cloud classification: https://lidar.univ-rennes.fr/en/3dmasc
| qAnimation              | PLUGIN_STANDARD_QANIMATION               | OFF           | Plugin to create videos: https://www.cloudcompare.org/doc/wiki/index.php/Animation_(plugin).
| qBroom                  | PLUGIN_STANDARD_QBROOM                   | OFF           | Interactive cloud cleaning tool: https://www.cloudcompare.org/doc/wiki/index.php/Virtual_broom_(plugin)
| qCanupo                 | PLUGIN_STANDARD_QCANUPO                  | OFF           | Automatic point cloud classification: https://www.cloudcompare.org/doc/wiki/index.php/CANUPO_(plugin)
| qCloudLayers            | PLUGIN_STANDARD_QCLOUDLAYERS             | OFF           | Manual point cloud classification/labelling: https://www.cloudcompare.org/doc/wiki/index.php/QCloudLayers_(plugin)
| qColorimetricSegmenter  | PLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER  | OFF           | Point cloud color-based segmentation: https://www.cloudcompare.org/doc/wiki/index.php/Colorimetric_Segmenter_(plugin)
| qCompass                | PLUGIN_STANDARD_QCOMPASS                 | OFF           | Digitization of geological structures and structural traces on point clouds: https://www.cloudcompare.org/doc/wiki/index.php/Compass_(plugin)
| qCork                   | PLUGIN_STANDARD_QCORK                    | OFF           | Mesh Boolean Operations: https://www.cloudcompare.org/doc/wiki/index.php/Cork_(plugin)
| qCSF                    | PLUGIN_STANDARD_QCSF                     | OFF           | Automatic ground/non-ground classification: https://www.cloudcompare.org/doc/wiki/index.php/CSF_(plugin)
| qFacets                 | PLUGIN_STANDARD_QFACETS                  | OFF           | Structural geology plugin: https://www.cloudcompare.org/doc/wiki/index.php/Facets_(plugin)
| qHoughNormals           | PLUGIN_STANDARD_QHOUGH_NORMALS           | OFF           | Normals computation: https://www.cloudcompare.org/doc/wiki/index.php/HoughNormals_(plugin)
| qHPR                    | PLUGIN_STANDARD_QHPR                     | OFF           | Hidden Point Removal: https://www.cloudcompare.org/doc/wiki/index.php/Hidden_Point_Removal_(plugin)
| qJSonRPCPlugin          | PLUGIN_STANDARD_QJSONRPC                 | OFF           | Json/RPC control plugin
| qM3C2                   | PLUGIN_STANDARD_QM3C2                    | OFF           | Robust point cloud distances computation: https://www.cloudcompare.org/doc/wiki/index.php/M3C2_(plugin)
| qMasonry                |                                          |               | Segmentation of masonry structures: https://www.cloudcompare.org/doc/wiki/index.php/Masonry_Segmentation_(plugin)
|  - qAutoSeg             | PLUGIN_STANDARD_MASONRY_QAUTO_SEG        | OFF           |
|  - qManualSeg           | PLUGIN_STANDARD_MASONRY_QMANUAL_SEG      | OFF           |
| qMeshBoolean            | PLUGIN_STANDARD_QMESH_BOOLEAN            | OFF           | Mesh Boolean Operations: https://www.cloudcompare.org/doc/wiki/index.php/Mesh_Boolean_(plugin)
| qMPlane                 | PLUGIN_STANDARD_QMPLANE                  | OFF           | Normal distance measurements against a defined plane: https://www.cloudcompare.org/doc/wiki/index.php/MPlane_(plugin)
| qPCL                    | PLUGIN_STANDARD_QPCL                     | OFF           | Interface to some algorithms of the PCL library: https://www.cloudcompare.org/doc/wiki/index.php/Point_Cloud_Library_Wrapper_(plugin)
| qPCV                    | PLUGIN_STANDARD_QPCV                     | OFF           | Ambient Occlusion for meshes or point clouds: https://www.cloudcompare.org/doc/wiki/index.php/ShadeVis_(plugin)
| qPoissonRecon           | PLUGIN_STANDARD_QPOISSON_RECON           | OFF           | Surface Mesh Reconstruction: https://www.cloudcompare.org/doc/wiki/index.php/Poisson_Surface_Reconstruction_(plugin)
| qRANSAC_SD              | PLUGIN_STANDARD_QRANSAC_SD               | OFF           | Automatic RANSAC shape detection: https://www.cloudcompare.org/doc/wiki/index.php/RANSAC_Shape_Detection_(plugin)
| qSRA                    | PLUGIN_STANDARD_QSRA                     | OFF           | Surface of Revolution Analysis: https://www.cloudcompare.org/doc/wiki/index.php/Surface_of_Revolution_Analysis_(plugin)
| qTreeIso                | PLUGIN_STANDARD_QTREEISO                 | OFF           | Individual Tree Isolation: https://www.cloudcompare.org/doc/wiki/index.php/Treeiso_(plugin)
| qPythonRuntime          | PLUGIN_PYTHON                            | OFF           | Python script runtime |
| qJSonRPCPlugin          | PLUGIN_STANDARD_QJSONRPC                 | OFF           | JSON-RPC server for AI agent integration |
| qSIBR                   | PLUGIN_STANDARD_QSIBR                    | OFF           | SIBR Gaussian Splatting viewers (**Linux/Windows**; CI and docs default **OFF on macOS**) |
| qDA3                    | PLUGIN_STANDARD_QDA3                     | OFF           | Depth Anything V3 — monocular depth, camera pose, COLMAP/GLB export, Automatic Reconstruction integration ([README](plugins/core/Standard/qDA3/README.md)). Requires `AICore_ENABLED=ON` (and `BUILD_RECONSTRUCTION=ON` for pipeline integration). |
| qFreeSplatter           | PLUGIN_STANDARD_QFREESPLATTER            | OFF           | FreeSplatter 3D Gaussian Splatting — uncalibrated photos to 3D Gaussians, pose recovery, SIBR-compatible PLY export, optional in-app viewer via qSIBR ([README](plugins/core/Standard/qFreeSplatter/README.md)). Requires `AICore_ENABLED=ON`; pair with `PLUGIN_STANDARD_QSIBR=ON` for visualization. |
| qLightGlue              | PLUGIN_STANDARD_QLIGHTGLUE               | OFF           | LightGlue sparse feature matching — ALIKED/SIFT descriptor pairs via GGUF, LGINP01 fixture I/O ([README](plugins/core/Standard/qLightGlue/README.md)). Requires `AICore_ENABLED=ON`. |

> 📖 **Plugin catalog:** [plugins/README.md](plugins/README.md) — per-plugin README index and AICore build recipes.

#### I/O Plugins

## IO Plugins

|       Plugin Name       |         CMake Option                     | Default Value | Description
|-------------------------|------------------------------------------|---------------|-------------
| qAdditionalIO           | PLUGIN_IO_QADDITIONAL                    | OFF           |
| qCoreIO                 | PLUGIN_IO_QCORE                          | ON            |
| qCSVMatrixIO            | PLUGIN_IO_QCSV_MATRIX                    | OFF           | Add support for CSV matrix files.
| qDraco                  | PLUGIN_IO_QDRACO                         | OFF           | Add support force draco files
| qE57IO                  | PLUGIN_IO_QE57                           | OFF           | Add support for e57 files using **libE57**.
| qFBXIO                  | PLUGIN_IO_QFBX                           | OFF           | Add support for AutoDesk FBX files using the official **FBX SDK**
| qLASFWIO                | PLUGIN_IO_QLAS_FWF                       | OFF           | Windows only. Support for LAS/LAZ with and without waveform using LIBlas (***deprecated, consider using qLASIO instead***).
| qLASIO                  | PLUGIN_IO_QLAS                           | OFF           | Support for LAS/LAZ with and without waveform (all platforms) using **LASZIP**.
| qPDALIO                 | PLUGIN_IO_QPDAL                          | OFF           | Add support for LAS/LAZ files using PDAL (***deprecated, consider using qLASIO instead***).
| qPhotoscanIO            | PLUGIN_IO_QPHOTOSCAN                     | OFF           |
| qRDBIO                  | PLUGIN_IO_QRDB                           | OFF           | Add support for RDB.
| qStepCADImport          | PLUGIN_IO_QSTEP                          | OFF           | Add support for STEP files.


> 📖 **For detailed plugin configuration**, see the platform-specific build guides.

---

## 🐛 Troubleshooting

### Common Issues

**Windows:**
- **DLL not found**: Ensure all DLLs are in PATH or installed directory
- **MSVC compiler errors**: Use Visual Studio 2022 or newer
- **Qt not found**: Set `QT_DIR` to your Qt installation

**Linux:**
- **Missing dependencies**: Install via `apt-get` or build from source
- **CUDA errors**: Ensure CUDA toolkit is properly installed
- **Qt plugin errors**: Install Qt platform plugins

**macOS:**
- **Code signing issues**: See macOS build guide for signing instructions
- **Architecture mismatch**: Build for your target architecture (x64/ARM64)

**For detailed troubleshooting**, refer to your platform's build guide linked above.

---

## 📦 Advanced Build Configuration

### Common Build Scenarios

#### Minimal Build (Core Only)

```bash
cmake -DBUILD_GUI=OFF \
      -DBUILD_PYTHON_MODULE=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_UNIT_TESTS=OFF \
      ..
cmake --build . --config Release
```

#### Full Featured Build (GUI + Python + CUDA)

```bash
cmake -DBUILD_CUDA_MODULE=ON \
      -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_GUI=ON \
      -DBUILD_RECONSTRUCTION=ON \
      -DPLUGIN_STANDARD_QJSONRPC=ON \
      -DPLUGIN_STANDARD_QSIBR=ON \
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
      -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
      ..
cmake --build . --config Release
```

#### AICore (qDA3 + qFreeSplatter + qLightGlue) Build

Builds `libAICore.so` (shared ggml inference core for DA3, FreeSplatter, and LightGlue) and the selected GUI plugins. Runtime **Auto** uses Metal → CPU on macOS and Vulkan → CPU on Linux/Windows by default. When `-DAICore_USE_CUDA=ON` and the CUDA backend is built, Auto becomes **CUDA → Vulkan → CPU** on Linux/Windows. SYCL remains explicit-only. CUDA is only enabled by `-DAICore_USE_CUDA=ON`; the unrelated CloudViewer `BUILD_CUDA_MODULE` option no longer adds CUDA to distributed AICore packages.

```bash
cmake -DBUILD_GUI=ON \
      -DBUILD_RECONSTRUCTION=ON \
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
      -DPLUGIN_STANDARD_QLIGHTGLUE=ON \
      ..
cmake --build . --config Release --target ACloudViewer
```

Release and CI builds require Vulkan explicitly:

```bash
cmake -DAICore_ENABLED=ON \
      -DAICore_USE_VULKAN=ON \
      ...
```

CPU-only build (no Vulkan SDK on the machine): `-DAICore_USE_VULKAN=OFF`.

`AICore_VULKAN_ENABLED` is the read-only outcome after dependency detection (same
pattern as `AICore_USE_CUDA` vs `AICore_CUDA_ENABLED`).

Vulkan development environment setup does not use Conda for the **system/pyenv**
Linux path. Use one platform script; all write the same env variable names.

| Platform | One-shot setup | Env file (source before `cmake`) |
|----------|----------------|----------------------------------|
| **Linux** | `util/install_deps_ubuntu.sh assume-yes` | `~/.local/share/acloudviewer/acloudviewer-vulkan-env.sh` |
| **Linux** (Vulkan only) | `util/install_vulkan_env.sh` | same |
| **macOS** | `util/install_vulkan_env.sh` | same path under `$HOME` |
| **Windows** | `.\util\install_vulkan_sdk_windows.ps1` | `%LOCALAPPDATA%\acloudviewer\acloudviewer-vulkan-env.ps1` |

Shared variables set by all scripts: `VULKAN_SDK`, `ACLOUDVIEWER_GLSLC`,
`ACLOUDVIEWER_SPIRV_HEADERS_DIR` (Linux SPIRV-Headers CMake package) or
`ACLOUDVIEWER_SPIRV_INCLUDE_DIR` (macOS/Windows SDK `Include`), and on Linux
`ACLOUDVIEWER_VULKAN_LIBRARY` (system `libvulkan.so`). CMake reads these
automatically; no manual `-DVulkan_*` flags after setup.

**Linux glslc note:** LunarG SDK **headers** are used on 20.04/22.04, but SDK
`bin/glslc` often requires GLIBC 2.34+ and fails on focal. `install_vulkan_linux.sh`
prefers SDK `glslc` when runnable, otherwise apt `glslc` or a pinned local shaderc
build (`~/.local/bin/glslc`).

**Conda Docker deps:** `Dockerfile_deps_conda` runs the same `install_vulkan_env.sh`
as other Linux images (with `ACLOUDVIEWER_UPDATE_BASHRC=0` in containers).

### Build-time vs runtime dependencies (all platforms)

| Component | Build machine | Installed app / wheel | End-user machine |
|-----------|---------------|----------------------|------------------|
| LunarG **Vulkan SDK** | Yes (or apt+scripts on Linux) | **No** | **No** |
| **`glslc`** | Yes | **No** | **No** |
| **SPIR-V / Vulkan headers** | Yes | **No** (shaders already embedded) | **No** |
| **`libggml-vulkan.so`** | Built locally | **Yes** (bundled) | **Yes** (from your build) |
| **GPU Vulkan driver / ICD** | Optional (for tests) | **No** | **Yes** (NVIDIA/AMD/Intel/Mesa) |
| **`libAICore.so`** | — | **Yes** | **Yes** |

**Can you copy the installer/wheel to another machine and run?** Yes, **if** the
target has a compatible OS/GPU driver and the same CPU architecture. The package
does **not** include the Vulkan SDK; it includes precompiled SPIR-V inside
`libggml-vulkan.so`. Without a working Vulkan ICD the app still runs on **CPU**
(Auto fallback). macOS production **Auto** uses Metal, not Vulkan.

```bash
# Linux / macOS
util/install_vulkan_env.sh
# or Linux full deps:
util/install_deps_ubuntu.sh assume-yes

# Windows (PowerShell)
.\util\install_vulkan_sdk_windows.ps1
```

Optional: skip shell profile hooks — Linux/macOS:
`ACLOUDVIEWER_UPDATE_BASHRC=0 util/install_vulkan_env.sh`; Windows:
`.\util\install_vulkan_sdk_windows.ps1 -SkipProfile`

The helper writes `acloudviewer-vulkan-env.sh` (or `.ps1` on Windows) and,
by default, installs a hook at the **top** of `~/.bashrc` and `~/.profile`
(PowerShell profile on Windows).

**Why `source ~/.bashrc` may not load Vulkan:** stock Ubuntu/Debian
`~/.bashrc` returns immediately for non-interactive shells (`case $- in *i*) ;;
*) return;; esac`). Hooks appended at the bottom never run when a script or IDE
runs `source ~/.bashrc`. Use one of:

```bash
source ~/.local/share/acloudviewer/acloudviewer-vulkan-env.sh   # recommended in scripts
# or open a new interactive terminal (login/interactive bash reads the top hook)
```

Re-run `util/install_vulkan_env.sh` to migrate an older bottom-of-bashrc hook.

Typical local configure/build after setup:

```bash
source ~/.local/share/acloudviewer/acloudviewer-vulkan-env.sh
cd build_app && cmake .. -DAICore_ENABLED=ON -DAICore_USE_VULKAN=ON
make -j"$(nproc)"
```

**GitHub CI (Linux / macOS / Windows):** workflows install the Vulkan SDK
(`install_vulkan_*` scripts). Build scripts pass `with_vulkan` to
`build_gui_app` / `build_pip_package` (maps to `-DAICore_USE_VULKAN=ON`; use
`without_vulkan` or `export AICore_USE_VULKAN=OFF` to disable). Linux Docker CI
runs `install_deps_ubuntu.sh`, which includes the same Vulkan setup.

Example:

```bash
source util/ci_utils.sh
build_gui_app with_conda package_installer with_vulkan
build_pip_package with_vulkan build_realsense build_jupyter
```

`glslc`, Vulkan headers, and SPIR-V headers are build-only dependencies. The
package contains the generated shaders and `ggml-vulkan` module, not the SDK or
validation layers. Linux/Windows users still need a Vulkan-capable display
driver. macOS production Auto remains Metal -> CPU; Vulkan/MoltenVK is built in
CI for compatibility testing and is selected only by explicit device name.

**Outputs:** `bin/libAICore.so`, private `libggml` core libraries,
`libggml-cpu` (required), optional backend modules, and the selected plugins.

**Models:**
- **DA3** GGUF: downloaded on first use to `~/cloudViewer_data/extract/da3_models` — [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf)
- **FreeSplatter** GGUF: auto-downloaded from [cloudViewer_downloads/3dgs](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs)

**Notes:**
- `AICore_ENABLED=ON` auto-enables `GGML_ENABLED`.
- ggml core and backends are private runtime files. Consumers never include or
  link ggml. `libAICore` has no hard CUDA dependency; `libggml-cuda` is loaded
  only when its driver and matching CUDA-major runtime are available.
- `libCV_DB_LIB` links AICore for the existing `ccImage` Qt adapter. COLMAP,
  pybind, qDA3, and qFreeSplatter otherwise use the public AICore API directly.
- CUDA is a developer backend: add `-DAICore_USE_CUDA=ON` explicitly.
- For **driver-only** CUDA installers (no CUDA Toolkit on target machines), also add
  `-DAICore_BUNDLE_CUDA_RUNTIME=ON` when packaging. Runtime libs land in
  `lib/cuda-runtime/`; `libcuda.so.1` still comes from the NVIDIA driver.
  GitHub CI **never** enables this by default.
- qFreeSplatter **Visualize** button requires `-DPLUGIN_STANDARD_QSIBR=ON` (Linux/Windows; **not supported on macOS** by default).
- **PostInstall** copies qSIBR runtime assets (`shaders/`, `sibr_resources/`, `ibr_resources.ini`) on **Linux and Windows only** when those directories exist in the build tree.

See [plugins/README.md](plugins/README.md), [qDA3 README](plugins/core/Standard/qDA3/README.md), and [qFreeSplatter README](plugins/core/Standard/qFreeSplatter/README.md).

#### DA3-only Build (legacy recipe)

Same as AICore build with only qDA3 enabled:

```bash
cmake -DBUILD_GUI=ON \
      -DBUILD_RECONSTRUCTION=ON \
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DBUILD_CUDA_MODULE=ON \
      ..
cmake --build . --config Release --target ACloudViewer
```

**Outputs:** `bin/libAICore.so`, `bin/libQDA3_PLUGIN.so`

#### Python Package Only

```bash
cmake -DBUILD_PYTHON_MODULE=ON \
      -DBUILD_GUI=OFF \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_PYTORCH_OPS=ON \
      ..
make python-package
make install-pip-package
```

#### AI Agent Development Build

```bash
cmake -DBUILD_GUI=ON \
      -DPLUGIN_STANDARD_QJSONRPC=ON \
      -DBUILD_PYTHON_MODULE=ON \
      ..
cmake --build . --config Release
```

### Building with Conda (Recommended)

```bash
# Create conda environment
conda env create -f .ci/conda_<platform>_cloudViewer.yml
conda activate cloudViewer

# Configure with conda
cmake -DBUILD_WITH_CONDA=ON \
      -DCONDA_PREFIX=$CONDA_PREFIX \
      ..

# Build
python scripts/build_<platform>.py
```

### Custom Installation Prefix

```bash
cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
cmake --build . --config Release --target install
```

### Parallel Build

```bash
# Windows
cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%

# Linux/macOS
cmake --build . --config Release --parallel $(nproc)
```

### Development vs Release Build

**Development Build** (default):
```bash
cmake -DDEVELOPER_BUILD=ON ..  # Adds git commit hash to version
```

**Release Build** (for distribution):
```bash
cmake -DDEVELOPER_BUILD=OFF \
      -DBUILD_COMMON_CUDA_ARCHS=ON \
      -DPACKAGE=ON \
      ..
cmake --build . --config Release --target install
```

> **Linux/Windows installer (`PACKAGE=ON`):** release packages contain
> Vulkan/CPU when their build tools are available. CUDA is excluded unless
> explicitly requested with `AICore_USE_CUDA=ON`. By default NVIDIA runtime
> libraries are **not** bundled (target machines need a matching CUDA Toolkit, or
> use Vulkan/CPU Auto). Opt-in driver-only CUDA: `-DAICore_BUNDLE_CUDA_RUNTIME=ON`
> (custom builds only; not GitHub CI). Vulkan users need only a compatible GPU
> driver/ICD.

**Custom CUDA installer (driver-only, not CI default):**

```bash
cmake -DDEVELOPER_BUILD=OFF \
      -DAICore_ENABLED=ON \
      -DAICore_USE_CUDA=ON \
      -DAICore_BUNDLE_CUDA_RUNTIME=ON \
      -DPACKAGE=ON \
      ..
cmake --build . --config Release --target install
```

Or via `ci_utils.sh`:

```bash
source util/ci_utils.sh
build_gui_app with_conda package_installer with_aicore_cuda bundle_cuda_runtime
```

Windows (`util/ci_utils.ps1` — same option names):

```powershell
. util\ci_utils.ps1
Build-GuiApp with_conda package_installer with_aicore_cuda bundle_cuda_runtime
```

Both helpers pass `-DAICore_USE_CUDA` and `-DAICore_BUNDLE_CUDA_RUNTIME` to CMake when those options are set.

Target machines need a compatible **NVIDIA driver** (provides `libcuda.so.1`) whose
version meets the bundled CUDA runtime major version. No CUDA Toolkit install required.

> **Python plugin (`PLUGIN_PYTHON=ON`):** By default only a **minimal** embedded runtime is installed (`PLUGIN_PYTHON_COPY_MINIMAL_ENV=ON`): Python stdlib + packages listed in `plugins/core/Standard/qPythonRuntime/requirements-release.txt`. It does **not** copy your entire pyenv/conda `site-packages` (which can be several GB if torch/Jupyter/etc. are installed). For release installers, point CMake at a **clean** interpreter:
>
> ```bash
> python -m venv /tmp/acloudviewer-python-pack
> /tmp/acloudviewer-python-pack/bin/pip install -r plugins/core/Standard/qPythonRuntime/requirements-release.txt
> cmake ... -DPython3_EXECUTABLE=/tmp/acloudviewer-python-pack/bin/python \
>             -DPLUGIN_PYTHON=ON -DPLUGIN_PYTHON_COPY_MINIMAL_ENV=ON \
>             -DPLUGIN_PYTHON_COPY_ENV=OFF
> ```
>
> To copy a larger dev environment (with bloat packages filtered), set `-DPLUGIN_PYTHON_COPY_ENV=ON`. To ship **no** bundled Python tree (embedded pycc/cccorelib only), set both `PLUGIN_PYTHON_COPY_ENV=OFF` and `PLUGIN_PYTHON_COPY_MINIMAL_ENV=OFF`.

> **macOS `.app` bundle (`lib_bundle_app.py --embed_python`):** Default **`--python_minimal`** embeds the same minimal set (stdlib + `requirements-release.txt`). NVIDIA CUDA runtime dylibs are **not** copied into `Frameworks/` (aligned with Linux/Windows packagers). Use `--python_full` only for local dev debugging with a complete conda env.

---

## 📚 Additional Documentation

- **[Windows Build Guide](docs/guides/compiling_doc/compiling-cloudviewer-windows.md)** - Detailed Windows build instructions
- **[Linux Build Guide](docs/guides/compiling_doc/compiling-cloudviewer-linux.md)** - Ubuntu/Debian build instructions
- **[macOS Build Guide](docs/guides/compiling_doc/compiling-cloudviewer-macos.md)** - macOS build instructions
- **[QUICKSTART.md](docs/guides/QUICKSTART.md)** - Comprehensive build reference
- **[Agent Integration](agent-integration/README.md)** - Building for AI agent development
- **[Plugin catalog](plugins/README.md)** - Per-plugin README index (AI, Standard, I/O)
- **[qDA3 Plugin](plugins/core/Standard/qDA3/README.md)** - Depth Anything V3 build, models, and Automatic Reconstruction integration
- **[qFreeSplatter Plugin](plugins/core/Standard/qFreeSplatter/README.md)** - FreeSplatter 3D Gaussian Splatting, models, and SIBR export
- **[qLightGlue Plugin](plugins/core/Standard/qLightGlue/README.md)** - LightGlue sparse feature matching (GGUF)

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for build requirements and development guidelines.

---

## 📝 Notes

### Platform-Specific Notes

**Windows:**
- Use Visual Studio 2022 for best compatibility
- Static runtime linking is recommended for distribution
- Qt 5.12 or Qt 6.2+ supported

**Linux:**
- Ubuntu 20.04+ recommended
- Install development packages for all dependencies
- Consider using system Qt for easier deployment

**macOS:**
- Universal binaries (ARM64) supported
- Code signing required for distribution
- Homebrew or Conda for dependency management

### Build Performance Tips

- Use parallel builds (`--parallel`)
- Enable Unity builds for faster compilation
- Use `ccache` or `sccache` for incremental builds
- Disable unused plugins to reduce build time
