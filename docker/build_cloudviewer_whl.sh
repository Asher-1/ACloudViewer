#!/bin/bash
set -euo pipefail

PACKAGE=${PACKAGE:-OFF}
DEVELOPER_BUILD=${DEVELOPER_BUILD:-OFF}
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-OFF}
BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE:-ON}
BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-ON}
BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-OFF}

export PACKAGE=${PACKAGE}
export DEVELOPER_BUILD=${DEVELOPER_BUILD}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
export BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}
export BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}
export BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}

export PYTHON_VERSION=$1
export NPROC=${NPROC:-$(($(nproc) + 2))} # run nproc+2 jobs to speed up the build
echo "PYTHON_VERSION: " python${PYTHON_VERSION}

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"
install_python_dependencies with-jupyter with-unit-test
build_pip_package build_realsense build_azure_kinect build_jupyter

# Note: Wheel testing is now done at runtime using test_wheel_runtime.sh
# This allows testing with GPU support (--gpus all) which is not available during build
# See: docker/test_wheel_runtime.sh and .github/workflows/ubuntu-wheel.yml

echo "Finish building cloudviewer wheel based on ${PYTHON_VERSION}!"
echo "mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}"
mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}
echo "Backup whl package to ${ACloudViewer_INSTALL}"

echo "Disk usage:"
df -h
echo
