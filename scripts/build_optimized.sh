#!/bin/bash
# CloudViewer Optimized Build Script
# This script builds CloudViewer with all performance optimizations enabled

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Default values
BUILD_TYPE="Release"
BUILD_DIR="build_optimized"
INSTALL_PREFIX="/usr/local"
NUM_CORES=$(nproc)
PYTHON_VERSION="3.8"
ENABLE_CUDA="OFF"
ENABLE_GUI="ON"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --install-prefix)
            INSTALL_PREFIX="$2"
            shift 2
            ;;
        --jobs)
            NUM_CORES="$2"
            shift 2
            ;;
        --python)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --enable-cuda)
            ENABLE_CUDA="ON"
            shift
            ;;
        --disable-gui)
            ENABLE_GUI="OFF"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --build-type TYPE    Build type (Release, Debug, RelWithDebInfo) [default: Release]"
            echo "  --build-dir DIR      Build directory [default: build_optimized]"
            echo "  --install-prefix DIR Install prefix [default: /usr/local]"
            echo "  --jobs N             Number of parallel jobs [default: nproc]"
            echo "  --python VERSION     Python version [default: 3.8]"
            echo "  --enable-cuda        Enable CUDA support"
            echo "  --disable-gui        Disable GUI components"
            echo "  --help               Show this help"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_status "Starting optimized CloudViewer build..."
print_status "Build type: $BUILD_TYPE"
print_status "Build directory: $BUILD_DIR"
print_status "Using $NUM_CORES parallel jobs"

# Check for required tools
print_status "Checking for required tools..."

command -v cmake >/dev/null 2>&1 || {
    print_error "cmake is required but not installed. Aborting."
    exit 1
}

command -v ninja >/dev/null 2>&1 && BUILD_GENERATOR="Ninja" || BUILD_GENERATOR="Unix Makefiles"
print_status "Using build generator: $BUILD_GENERATOR"

# Check CMake version
CMAKE_VERSION=$(cmake --version | head -n1 | cut -d' ' -f3)
print_status "CMake version: $CMAKE_VERSION"

# Check for compiler optimizations support
if command -v gcc >/dev/null 2>&1; then
    GCC_VERSION=$(gcc --version | head -n1)
    print_status "GCC version: $GCC_VERSION"
elif command -v clang >/dev/null 2>&1; then
    CLANG_VERSION=$(clang --version | head -n1)
    print_status "Clang version: $CLANG_VERSION"
fi

# Create build directory
print_status "Creating build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure CMake with performance optimizations
print_status "Configuring CMake with performance optimizations..."

CMAKE_ARGS=(
    -G "$BUILD_GENERATOR"
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    
    # Performance optimization flags
    -DENABLE_PERFORMANCE_OPTIMIZATION=ON
    -DENABLE_LTO=ON
    -DENABLE_NATIVE_ARCH=ON
    -DUSE_SIMD=ON
    
    # Enable parallelization
    -DWITH_OPENMP=ON
    -DBUILD_CUDA_MODULE="$ENABLE_CUDA"
    
    # Enable advanced features
    -DBUILD_GUI="$ENABLE_GUI"
    -DBUILD_WEBRTC=ON
    -DBUILD_PYTHON_MODULE=ON
    -DBUILD_EXAMPLES=OFF  # Disable to reduce build time
    -DBUILD_UNIT_TESTS=OFF  # Disable to reduce build time
    
    # Memory optimization
    -DSTATIC_WINDOWS_RUNTIME=OFF
    -DGLIBCXX_USE_CXX11_ABI=ON
    
    # Bundle optimization
    -DBUNDLE_CLOUDVIEWER_ML=OFF  # Disable to reduce package size
    
    # System libraries (prefer system versions for better optimization)
    -DUSE_SYSTEM_EIGEN3=ON
    -DUSE_SYSTEM_FLANN=ON
    -DUSE_SYSTEM_GLEW=ON
    -DUSE_SYSTEM_GLFW=ON
    -DUSE_SYSTEM_JPEG=ON
    -DUSE_SYSTEM_PNG=ON
    -DUSE_SYSTEM_TBB=ON
)

# Add CUDA-specific optimizations if enabled
if [ "$ENABLE_CUDA" = "ON" ]; then
    print_status "Enabling CUDA optimizations..."
    CMAKE_ARGS+=(
        -DBUILD_COMMON_CUDA_ARCHS=ON
        -DCUDA_ARCH_BIN="6.0;6.1;7.0;7.5;8.0;8.6"
    )
fi

# Set compiler-specific optimizations
if [ "$BUILD_TYPE" = "Release" ]; then
    export CFLAGS="-O3 -march=native -mtune=native -DNDEBUG"
    export CXXFLAGS="-O3 -march=native -mtune=native -DNDEBUG"
    export LDFLAGS="-Wl,-O1 -Wl,--as-needed"
    
    # Enable LTO if supported
    if gcc -v --help 2>/dev/null | grep -q "flto" || clang -v --help 2>/dev/null | grep -q "flto"; then
        export CFLAGS="$CFLAGS -flto"
        export CXXFLAGS="$CXXFLAGS -flto"
        export LDFLAGS="$LDFLAGS -flto"
        print_status "LTO enabled in compiler flags"
    fi
fi

# Run CMake configuration
print_status "Running CMake configuration..."
cmake "${CMAKE_ARGS[@]}" ..

if [ $? -ne 0 ]; then
    print_error "CMake configuration failed!"
    exit 1
fi

# Build the project
print_status "Building CloudViewer with $NUM_CORES parallel jobs..."

if [ "$BUILD_GENERATOR" = "Ninja" ]; then
    ninja -j"$NUM_CORES"
else
    make -j"$NUM_CORES"
fi

if [ $? -ne 0 ]; then
    print_error "Build failed!"
    exit 1
fi

# Run performance benchmark if requested
if [ -f "../scripts/performance_benchmark.py" ]; then
    print_status "Running performance benchmark..."
    python3 ../scripts/performance_benchmark.py --quick --output "benchmark_optimized.json"
fi

# Install if requested
if [ "${INSTALL_CLOUDVIEWER:-false}" = "true" ]; then
    print_status "Installing CloudViewer..."
    if [ "$BUILD_GENERATOR" = "Ninja" ]; then
        ninja install
    else
        make install
    fi
fi

# Create Python wheel if Python module was built
if [ -d "lib/python_package" ]; then
    print_status "Creating optimized Python wheel..."
    cd lib/python_package
    
    # Set wheel optimization flags
    export CFLAGS="-O3 -march=native -DNDEBUG"
    export CXXFLAGS="-O3 -march=native -DNDEBUG"
    
    python3 setup.py bdist_wheel
    
    # Check wheel size
    WHEEL_FILE=$(find dist -name "*.whl" | head -n1)
    if [ -f "$WHEEL_FILE" ]; then
        WHEEL_SIZE=$(du -h "$WHEEL_FILE" | cut -f1)
        print_status "Python wheel created: $WHEEL_FILE (size: $WHEEL_SIZE)"
    fi
    
    cd ../..
fi

# Print build summary
print_status "Build completed successfully!"
print_status "Build directory: $(pwd)"
print_status "Build type: $BUILD_TYPE"

if [ -f "compile_commands.json" ]; then
    print_status "Compile commands available for analysis"
fi

# Show optimization summary
print_status "Applied optimizations:"
echo "  ✓ Compiler optimization level: -O3"
echo "  ✓ Target-specific optimization: -march=native"
echo "  ✓ SIMD instructions enabled"
echo "  ✓ Link-time optimization (LTO)"
echo "  ✓ OpenMP parallelization"
echo "  ✓ Bundle size optimization"

if [ "$ENABLE_CUDA" = "ON" ]; then
    echo "  ✓ CUDA acceleration enabled"
fi

print_status "To install: make install (or ninja install)"
print_status "To test: ctest"
print_status "To benchmark: python3 ../scripts/performance_benchmark.py"

# Final recommendations
print_warning "Recommendations for best performance:"
echo "  - Set OMP_NUM_THREADS=$(nproc) for optimal OpenMP performance"
echo "  - Use the Release build for production deployments"
echo "  - Consider using jemalloc or tcmalloc for better memory performance"
echo "  - Profile your specific workload for application-specific optimizations"

print_status "Build script completed!"