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
| `GGML_ENABLED` | OFF | Build ggml inference library (auto-enabled when `AICore_ENABLED=ON`) |
| `GGML_USE_CUDA` | OFF | Enable ggml CUDA backend on Linux/Windows (also on when `BUILD_CUDA_MODULE=ON`) |
| `GGML_USE_VULKAN` | OFF (all platforms) | Opt-in only; disabled by default (Vulkan deployment is not supported) |
| `GGML_USE_OPENCL` | Linux/Win: ON, macOS: OFF | Auto-detect OpenCL 3.0 headers + ICD + Python3 when ON; **not built on macOS** |
| `GGML_USE_METAL` | Apple: ON, else OFF | ggml Metal backend (Apple only) |

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
| qSIBR                   | PLUGIN_STANDARD_QSIBR                    | OFF           | SIBR Gaussian Splatting viewers |
| qDA3                    | PLUGIN_STANDARD_QDA3                     | OFF           | Depth Anything V3 — monocular depth, camera pose, COLMAP/GLB export, Automatic Reconstruction integration ([README](plugins/core/Standard/qDA3/README.md)). Requires `AICore_ENABLED=ON` (and `BUILD_RECONSTRUCTION=ON` for pipeline integration). |
| qFreeSplatter           | PLUGIN_STANDARD_QFREESPLATTER            | OFF           | FreeSplatter 3D Gaussian Splatting — uncalibrated photos to 3D Gaussians, pose recovery, SIBR-compatible PLY export, optional in-app viewer via qSIBR ([README](plugins/core/Standard/qFreeSplatter/README.md)). Requires `AICore_ENABLED=ON`; pair with `PLUGIN_STANDARD_QSIBR=ON` for visualization. |

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
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
      -DPLUGIN_STANDARD_QJSONRPC=ON \
      -DPLUGIN_STANDARD_QSIBR=ON \
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
      ..
cmake --build . --config Release
```

#### AICore (qDA3 + qFreeSplatter) Build

Builds `libAICore.so` (shared ggml inference core for DA3 and FreeSplatter) and the selected GUI plugins. Multiple ggml backends can be compiled into one build; runtime **Auto** selects CUDA → OpenCL → Vulkan → CPU (first available). CUDA requires `BUILD_CUDA_MODULE=ON` or `-DGGML_USE_CUDA=ON`; Vulkan/OpenCL are auto-detected when dependencies are present.

```bash
cmake -DBUILD_GUI=ON \
      -DBUILD_RECONSTRUCTION=ON \
      -DAICore_ENABLED=ON \
      -DPLUGIN_STANDARD_QDA3=ON \
      -DPLUGIN_STANDARD_QFREESPLATTER=ON \
      -DBUILD_CUDA_MODULE=ON \
      ..
cmake --build . --config Release --target ACloudViewer
```

**Outputs:** `bin/libAICore.so`, `bin/libQDA3_PLUGIN.so`, `bin/libQFREESPLATTER_PLUGIN.so` (when enabled)

**Models:**
- **DA3** GGUF: downloaded on first use to `~/cloudViewer_data/extract/da3_models` — [mudler/depth-anything.cpp-gguf](https://huggingface.co/mudler/depth-anything.cpp-gguf)
- **FreeSplatter** GGUF: auto-downloaded from [cloudViewer_downloads/3dgs](https://github.com/Asher-1/cloudViewer_downloads/releases/tag/3dgs)

**Notes:**
- `AICore_ENABLED=ON` auto-enables `GGML_ENABLED`.
- With CUDA, `libAICore.so` is large (~200–250 MB) because `libggml-cuda.a` CUDA kernels for `CMAKE_CUDA_ARCHITECTURES` are linked statically into AICore only. **`libCV_DB_LIB.so` does not link AICore** — use `aicore::depth::ImageDepth` from `aicore/depth_image.h` in code that already links `libAICore.so`.
- CPU-only AICore: omit `-DBUILD_CUDA_MODULE=ON`.
- qFreeSplatter **Visualize** button requires `-DPLUGIN_STANDARD_QSIBR=ON` (not supported on macOS by default).

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

> **Linux installer (`PACKAGE=ON`):** `pack_ubuntu.sh` does **not** bundle NVIDIA CUDA runtime libraries (`libcublas`, `libcublasLt`, `libcudart`, etc.) — same as the Windows packager (`pack_windows.ps1` already filtered CUDA DLLs before the Linux fix). GPU / AICore features require an NVIDIA driver and a **matching CUDA toolkit/runtime** on the target machine.

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
