# CloudViewer Unit Tests

This directory contains C++ unit tests for the CloudViewer library.

## Structure

```
tests/
├── camera/          # Camera parameter tests
├── core/            # Core functionality tests (Tensor, Device, etc.)
├── data/            # Dataset loading tests
├── geometry/        # Geometry tests (PointCloud, Mesh, etc.)
├── io/              # I/O tests (file format reading/writing)
├── ml/              # Machine learning tests
├── pipelines/       # Pipeline tests (registration, odometry, etc.)
├── t/               # Tensor-based API tests
│   ├── geometry/    # Tensor geometry tests
│   ├── io/          # Tensor I/O tests
│   └── pipelines/   # Tensor pipeline tests
├── test_utility/    # Test utilities (Compare, Rand, Sort, etc.)
├── utility/         # Utility function tests
├── visualization/   # Visualization tests
├── CMakeLists.txt   # Build configuration
├── Main.cpp         # Test runner main entry point
├── Tests.cpp        # Test helper implementations
└── UnitTest.h       # Test utilities and macros
```

## Building Tests

Tests are built when `BUILD_UNIT_TESTS` is enabled:

```bash
cmake -B build \
    -DBUILD_UNIT_TESTS=ON \
    -DBUILD_CUDA_MODULE=ON \  # Optional: for CUDA tests
    -DCMAKE_BUILD_TYPE=Release

cmake --build build -j$(nproc)
```

## Running Tests

### Run All Tests

```bash
cd build
./bin/tests
```

### Run with Options

```bash
# Shuffle test order (randomized)
./bin/tests --gtest_shuffle

# Run specific test suite
./bin/tests --gtest_filter=Core*

# Run specific test
./bin/tests --gtest_filter=Tensor.Constructor

# Exclude tests
./bin/tests --gtest_filter=-*CUDA*

# Repeat tests
./bin/tests --gtest_repeat=10

# List all tests
./bin/tests --gtest_list_tests

# Set random seed for reproducibility
./bin/tests --gtest_shuffle --gtest_random_seed=12345

# Disable P2P for CUDA tests
./bin/tests --disable_p2p
```

### Run with Verbose Output

```bash
# Show all output
./bin/tests --gtest_print_time=1

# Break on failure
./bin/tests --gtest_break_on_failure

# Generate XML report
./bin/tests --gtest_output=xml:test_results.xml
```

## Test Categories

### Core Tests
- **Blob**: Blob data structure tests
- **Device**: Device management (CPU/CUDA/SYCL)
- **Tensor**: Tensor operations and manipulations
- **HashMap**: Hash map data structure
- **Linalg**: Linear algebra operations
- **ParallelFor**: Parallel execution tests

### Geometry Tests
- **PointCloud**: Point cloud operations
- **TriangleMesh**: Mesh operations
- **Image**: Image processing
- **Octree**: Octree data structure
- **KDTree**: KD-tree search

### I/O Tests
- File format tests (PCD, PLY, STL, OBJ, etc.)
- Image I/O (PNG, JPG)
- Sensor I/O (Azure Kinect)

### Pipeline Tests
- **Registration**: Point cloud registration (ICP, etc.)
- **Odometry**: Visual odometry
- **Integration**: TSDF integration
- **SLAC**: Simultaneous localization and calibration

## Conditional Compilation

Tests support conditional compilation based on build options:

- `BUILD_CUDA_MODULE`: Enables CUDA-specific tests
- `BUILD_AZURE_KINECT`: Enables Azure Kinect sensor tests
- `BUILD_GUI`: Enables visualization rendering tests
- `WITH_IPP`: Enables Intel IPP tests

Tests that require unavailable features are automatically disabled with the `DISABLED_` prefix.

## Test Utilities

### Compare Utilities
```cpp
using namespace cloudViewer::tests;

// Compare vectors
ExpectEQ(vec1, vec2);

// Compare with tolerance
EXPECT_NEAR(val1, val2, 1e-5);

// Compare Eigen matrices
ExpectEQ(mat1, mat2);
```

### Random Data Generation
```cpp
// Generate random point cloud
auto pcd = Rand::RandPointCloud(1000);

// Generate random mesh
auto mesh = Rand::RandTriangleMesh();

// Generate random tensor
auto tensor = Rand::RandTensor({100, 3});
```

### Tensor Comparison
```cpp
// Compare tensors with tolerance
AllCloseOrShow(tensor1, tensor2, rtol, atol);
```

## Adding New Tests

### 1. Create Test File

```cpp
// tests/geometry/MyGeometry.cpp
#include "tests/UnitTest.h"
#include "cloudViewer/geometry/MyGeometry.h"

namespace cloudViewer {
namespace tests {

TEST(MyGeometry, Constructor) {
    geometry::MyGeometry geom;
    EXPECT_TRUE(geom.IsEmpty());
}

TEST(MyGeometry, Operations) {
    // Your test code
}

}  // namespace tests
}  // namespace cloudViewer
```

### 2. The test will be automatically included

The `CMakeLists.txt` uses `GLOB_RECURSE` to automatically find all `*.cpp` files.

### 3. Build and Run

```bash
cmake --build build
./build/bin/tests --gtest_filter=MyGeometry*
```

## CI/CD Integration

Tests are integrated with CI/CD through `util/ci_utils.sh`:

```bash
# From build directory
source ../util/ci_utils.sh

# Run C++ tests
run_cpp_unit_tests

# Run all tests (C++ + Python)
run_all_tests
```

## Troubleshooting

### Test Fails with CUDA Errors

```bash
# Disable P2P transfer
./bin/tests --disable_p2p

# Run only CPU tests
./bin/tests --gtest_filter=-*CUDA*
```

### Memory Issues

```bash
# Set memory limit
export LOW_MEM_USAGE=ON
./bin/tests --gtest_filter=-*Reduce*Sum*
```

### Debugging Failed Tests

```bash
# Run with GDB
gdb --args ./bin/tests --gtest_filter=FailingTest

# Run with Valgrind
valgrind --leak-check=full ./bin/tests --gtest_filter=FailingTest
```

## References

- [GoogleTest Documentation](https://google.github.io/googletest/)
- [CloudViewer Documentation](https://asher-1.github.io/ACloudViewer/)
- [Testing Guide](../../../docker/TESTING.md)
