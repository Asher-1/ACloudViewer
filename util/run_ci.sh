#!/usr/bin/env bash
set -euo pipefail

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source "$(dirname "$0")"/ci_utils.sh

echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

if [ "$BUILD_CUDA_MODULE" == "ON" ] &&
    ! nvcc --version | grep -q "release ${CUDA_VERSION[1]}" 2>/dev/null; then
    install_cuda_toolkit with-cudnn purge-cache
    nvcc --version
fi

if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
    install_python_dependencies with-unit-test with-cuda purge-cache
else
    install_python_dependencies with-unit-test purge-cache
fi

echo "using python: $(which python)"
python --version
echo -n "Using pip: "
python -m pip --version
echo -n "Using pytest:"
python -m pytest --version
echo "using cmake: $(which cmake)"
cmake --version

build_all

echo "Building examples iteratively..."
make VERBOSE=1 -j"$NPROC" build-examples-iteratively
echo

echo "running CloudViewer C++ unit tests..."
run_cpp_unit_tests
echo "try importing CloudViewer Python package"
test_wheel
echo "running CloudViewer Python tests..."
run_python_tests
echo

echo "Test building a C++ example with installed CloudViewer..."
test_cpp_example "${runExample:=ON}"
echo

echo "test uninstalling CloudViewer..."
make uninstall
