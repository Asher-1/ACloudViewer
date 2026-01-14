# Python Unit Test

## Installation

Install test dependencies:

```bash
pip install -r python/requirements_test.txt
```

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest python/test/

# Run specific test file
pytest python/test/test_color_map.py

# Run specific test function
pytest python/test/test_color_map.py::test_color_map

# Use -s to show stdout
pytest -s python/test/

# Use -v for verbose output
pytest -v python/test/
```

### Selective Testing

```bash
# Skip ML ops tests
pytest python/test/ --ignore=python/test/ml_ops/

# Skip slow tests
pytest python/test/ -m "not slow"

# Skip GPU tests
pytest python/test/ -m "not gpu"

# Run only specific markers
pytest python/test/ -m "ml_ops"
```

### CI/CD Testing

Use the provided CI utility script:

```bash
# From the project root, after building
cd build_app
source ../util/ci_utils.sh
run_python_tests
```

## Test Structure

The test suite is organized as follows:

- `core/` - Core functionality tests (Tensor, Device, etc.)
- `io/` - I/O operations tests
- `ml/` - Machine learning contrib tests
- `ml_ops/` - ML operations tests (requires ML modules built)
- `t/` - Tensor-based API tests
- `visualization/` - Visualization tests
- `data/` - Data loading tests

## Notes

- Some tests require CUDA support and will be skipped if CUDA is not available
- ML ops tests require BUILD_PYTORCH_OPS or BUILD_TENSORFLOW_OPS to be ON
- Tests use pytest markers to categorize and selectively run tests
- The test suite supports both CPU and GPU devices automatically
