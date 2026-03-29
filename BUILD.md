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

| Plugin | Description |
|--------|-------------|
| `PLUGIN_STANDARD_QJSONRPC` | JSON-RPC server for AI agent integration |
| `PLUGIN_STANDARD_QSIBR` | SIBR Gaussian Splatting viewers |
| `PLUGIN_STANDARD_QPCL` | PCL integration (point cloud algorithms) |
| `PLUGIN_STANDARD_QANIMATION` | Animation and video export (requires FFmpeg) |
| `PLUGIN_STANDARD_QPOISSON_RECON` | Poisson surface reconstruction |
| `PLUGIN_STANDARD_QRANSAC_SD` | RANSAC shape detection |
| `PLUGIN_STANDARD_QFACETS` | Facet segmentation |
| `PLUGIN_STANDARD_QHPR` | Hidden Point Removal |
| `PLUGIN_STANDARD_QSRA` | Surface of Revolution Analysis |
| `PLUGIN_STANDARD_QCORK` | Cork boolean operations |
| `PLUGIN_STANDARD_QCANUPO` | CANUPO classification |
| `PLUGIN_STANDARD_QPYTHONRUNTIME` | Python script runtime |

#### I/O Plugins

| Plugin | Description |
|--------|-------------|
| `PLUGIN_IO_QFBX` | Autodesk FBX file format support |
| `PLUGIN_IO_QE57` | E57 point cloud format |
| `PLUGIN_IO_QLAS` | LAS/LAZ point cloud format |
| `PLUGIN_IO_QPDAL` | PDAL integration (multiple formats) |
| `PLUGIN_IO_QPHOTOSCAN` | Agisoft Photoscan format |
| `PLUGIN_IO_QRDB` | Riegl RDB format |
| `PLUGIN_IO_QDRACO` | Google Draco compressed format |
| `PLUGIN_IO_QCSV_MATRIX` | CSV matrix files |

#### GL Plugins

| Plugin | Description |
|--------|-------------|
| `PLUGIN_GL_QSSAO` | Screen Space Ambient Occlusion |
| `PLUGIN_GL_QEDL` | Eye Dome Lighting |

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
      ..
cmake --build . --config Release
```

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

---

## 📚 Additional Documentation

- **[Windows Build Guide](docs/guides/compiling_doc/compiling-cloudviewer-windows.md)** - Detailed Windows build instructions
- **[Linux Build Guide](docs/guides/compiling_doc/compiling-cloudviewer-linux.md)** - Ubuntu/Debian build instructions
- **[macOS Build Guide](docs/guides/compiling_doc/compiling-cloudviewer-macos.md)** - macOS build instructions
- **[QUICKSTART.md](docs/guides/QUICKSTART.md)** - Comprehensive build reference
- **[Agent Integration](agent-integration/README.md)** - Building for AI agent development

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
