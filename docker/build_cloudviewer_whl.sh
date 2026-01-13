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

# Auto-detect Qt version and set USE_QT6 accordingly
# Check if USE_QT6 is already set via environment variable
if [ -z "${USE_QT6:-}" ]; then
    # Try to detect Qt version from system
    # First check if Qt6 is available
    if command -v qmake6 &> /dev/null; then
        QT_VERSION=$(qmake6 -query QT_VERSION 2>/dev/null || echo "")
    elif [ -f "/usr/lib/x86_64-linux-gnu/cmake/Qt6/Qt6Config.cmake" ]; then
        QT_VERSION="6.x"
    elif [ -d "/opt/qt6" ]; then
        # Check for aqtinstall-based Qt6
        QT_VERSION="6.x"
    elif command -v qmake &> /dev/null; then
        QT_VERSION=$(qmake -query QT_VERSION 2>/dev/null || echo "")
    else
        QT_VERSION=""
    fi
    
    # Determine USE_QT6 based on detected version
    if [[ "$QT_VERSION" == 6* ]]; then
        export USE_QT6=ON
        echo "Detected Qt6 (version: $QT_VERSION), setting USE_QT6=ON"
    else
        export USE_QT6=OFF
        echo "Detected Qt5 or unknown (version: $QT_VERSION), setting USE_QT6=OFF"
    fi
else
    echo "USE_QT6 already set to: ${USE_QT6}"
fi

# Source Qt6 environment if available (for focal with aqtinstall)
if [ -f "/etc/profile.d/qt6.sh" ]; then
    echo "Sourcing Qt6 environment from /etc/profile.d/qt6.sh"
    source /etc/profile.d/qt6.sh
fi

export USE_QT6=${USE_QT6}

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
