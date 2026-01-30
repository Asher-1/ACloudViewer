# Build from Source

Complete guide for building ACloudViewer from source code on different platforms.

## 📋 Platform-Specific Guides

- [**Linux Building Guide**](compiling_doc/compiling-cloudviewer-linux.md)
- [**macOS Building Guide**](compiling_doc/compiling-cloudviewer-macos.md)
- [**Windows Building Guide**](compiling_doc/compiling-cloudviewer-windows.md)

## 💻 System Requirements

### Operating Systems

| Platform | Version | Compiler |
|----------|---------|----------|
| Ubuntu | 20.04+ | GCC 9+, Clang 10+ |
| macOS | 10.14+ | XCode 8.0+ |
| Windows | 10 (64-bit) | Visual Studio 2022+ |

### Build Tools

**CMake 3.19+**

- Ubuntu 20.04+:
  ```bash
  # Use default OS repository
  sudo apt-get install cmake
  ```

- macOS:
  ```bash
  brew install cmake
  ```

- Windows:
  - Download from [CMake download page](https://cmake.org/download/)

### Optional: GPU Support

**CUDA 11.0+** (for GPU acceleration)

ACloudViewer supports GPU acceleration through CUDA on Linux. See the [official CUDA installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## 📦 Getting the Source Code

```bash
# Clone the repository
git clone https://github.com/Asher-1/ACloudViewer.git
cd ACloudViewer
```

## 🔧 Quick Build Instructions

### Ubuntu/Linux

```bash
# Install dependencies
./util/install_deps_ubuntu.sh

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)

# Install (optional)
sudo make install
```

### macOS

```bash
# Install dependencies
brew install qt5 eigen boost glew

# Build
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/qt@5/lib/cmake \
      ..
make -j$(sysctl -n hw.ncpu)
```

### Windows

```bat
REM Build
mkdir build && cd build

cmake -G "Visual Studio 17 2022" -A x64 ^
      -DCMAKE_INSTALL_PREFIX="<cloudViewer_install_directory>" ^
      ..

cmake --build . --config Release --target ALL_BUILD

REM Install (optional)
cmake --build . --config Release --target INSTALL
```

## 🐍 Building Python Package

### Setup Python Environment

```bash
# Create virtual environment
python3 -m venv cloudViewer_env
source cloudViewer_env/bin/activate  # On Windows: cloudViewer_env\Scripts\activate

# Install build dependencies
pip install -r python/requirements_build.txt
```

### Build Python Wheel

```bash
mkdir build && cd build

# Configure with Python bindings
cmake -DBUILD_PYTHON_MODULE=ON \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Build
make pip-package -j$(nproc)

# Install
pip install lib/python_package/pip_package/cloudviewer*.whl
```

### Verify Installation

```bash
python -c "import cloudViewer; print(cloudViewer.__version__)"
```

## ⚙️ Common Build Options

### Core Options

```bash
cmake \
  -DCMAKE_BUILD_TYPE=Release \          # Release, Debug, RelWithDebInfo
  -DBUILD_SHARED_LIBS=OFF \            # Static libraries
  -DDEVELOPER_BUILD=OFF \              # Developer mode
  -DCMAKE_INSTALL_PREFIX=/usr/local \  # Installation directory
  ..
```

### Python Options

```bash
cmake \
  -DBUILD_PYTHON_MODULE=ON \           # Enable Python bindings
  -DBUNDLE_CLOUDVIEWER_ML=ON \         # Include ML module
  -DCLOUDVIEWER_ML_ROOT=/path/to/ml \  # ML module path
  ..
```

### GPU and ML Options

```bash
cmake \
  -DBUILD_CUDA_MODULE=ON \             # Enable CUDA
  -DBUILD_PYTORCH_OPS=ON \             # Build PyTorch ops
  -DBUILD_TENSORFLOW_OPS=OFF \         # Build TensorFlow ops
  ..
```

### Feature Options

```bash
cmake \
  -DWITH_OPENMP=ON \                   # Enable OpenMP
  -DWITH_IPP=ON \                      # Intel IPP support
  -DBUILD_GUI=ON \                     # Build GUI application
  -DBUILD_RECONSTRUCTION=ON \          # Reconstruction module
  ..
```

## 📚 Advanced Topics

### OpenMP Support

OpenMP greatly accelerates computation on multi-core CPUs. ACloudViewer automatically detects OpenMP support when `WITH_OPENMP=ON`.

**macOS Note**: The default LLVM compiler doesn't support OpenMP. Use GCC instead:

```bash
brew install gcc
cmake -DCMAKE_C_COMPILER=gcc-12 \
      -DCMAKE_CXX_COMPILER=g++-12 \
      -DWITH_OPENMP=ON \
      ..
```

### ML Module

To build the complete ML module with models and pipelines:

```bash
cmake \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_PYTORCH_OPS=ON \
  -DBUILD_TENSORFLOW_OPS=OFF \
  -DBUNDLE_CLOUDVIEWER_ML=ON \
  -DCLOUDVIEWER_ML_ROOT=https://github.com/intel-isl/CloudViewer-ML.git \
  ..

make -j$(nproc) install-pip-package
```

**Important**: PyTorch and TensorFlow Python wheels on Linux have different CXX ABIs. Official Linux wheels only support PyTorch, not TensorFlow.

### CUDA Support

```bash
# Check CUDA installation
nvidia-smi  # Verify CUDA-enabled GPU
nvcc -V     # Verify CUDA compiler

# Build with CUDA
cmake \
  -DBUILD_CUDA_MODULE=ON \
  -DBUILD_COMMON_CUDA_ARCHS=ON \
  ..
```

## 🧪 Running Tests

### C++ Unit Tests

```bash
cmake -DBUILD_UNIT_TESTS=ON ..
make -j$(nproc)
./bin/tests
```

### Python Unit Tests

```bash
# Activate virtualenv
source cloudViewer_env/bin/activate

# Install test dependencies
pip install pytest

# Install package
make install-pip-package

# Run tests
pytest ../python/test
```

## 📤 Publishing Python Wheels

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*
```

## 🔗 Additional Resources

- [Linux Build Guide](compiling_doc/compiling-cloudviewer-linux.md)
- [macOS Build Guide](compiling_doc/compiling-cloudviewer-macos.md)
- [Windows Build Guide](compiling_doc/compiling-cloudviewer-windows.md)
- [Documentation Build Guide](../automation/BUILD_DOCUMENTATION.md)
- [CI/CD Guide](CI_DOCUMENTATION_GUIDE.md)

## ⚠️ Notes

### Custom Modifications

The following third-party libraries have been customized:
- `3rdparty/rply`
- `libs/Reconstruction/lib/PoissonRecon`

### Platform-Specific Issues

- **macOS**: OpenMP requires custom compiler installation
- **Windows**: Requires Visual Studio 2022+ for C++17 support

---

**Last Updated**: 2026-01-13  
**Maintained by**: ACloudViewer Team
