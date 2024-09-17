#!/usr/bin/env bash
set -euo pipefail

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source "$(dirname "$0")"/ci_utils.sh

echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

if [ "$BUILD_CUDA_MODULE" == "ON" ] &&
    ! nvcc --version 2>/dev/null | grep -q "release ${CUDA_VERSION[1]}"; then
    install_cuda_toolkit with-cudnn purge-cache
    nvcc --version
fi

if [ "$BUILD_CUDA_MODULE" == "ON" ]; then
    install_python_dependencies with-unit-test with-cuda purge-cache
else
    install_python_dependencies with-unit-test purge-cache
fi

build_all

echo "Building examples iteratively..."
make VERBOSE=1 -j"$NPROC" build-examples-iteratively
echo

df -h

echo "Running CloudViewer C++ unit tests..."
run_cpp_unit_tests

# Run on GPU only. CPU versions run on Github already
if nvidia-smi >/dev/null 2>&1; then
    echo "Try importing CloudViewer Python package"
    test_wheel lib/python_package/pip_package/CloudViewer*.whl
    df -h
    echo "Running CloudViewer Python tests..."
    run_python_tests
    echo
fi

echo "Test building a C++ example with installed CloudViewer..."
test_cpp_example "${runExample:=ON}"
echo

echo "Test uninstalling CloudViewer..."
make uninstall
