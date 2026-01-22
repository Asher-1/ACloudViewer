#!/bin/bash
set -euo pipefail

PACKAGE=${PACKAGE:-ON}
DEVELOPER_BUILD=${DEVELOPER_BUILD:-OFF}
BUILD_CUDA_MODULE_FLAG=${BUILD_CUDA_MODULE:-ON}
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-OFF}
BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-OFF}
BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-OFF}

export PACKAGE=${PACKAGE}
export DEVELOPER_BUILD=${DEVELOPER_BUILD}
export BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}
export BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}
export BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}

export PYTHON_VERSION=$1
export ONLY_BUILD_CUDA=$2
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

# for python plugin
# you can use PackageManager to install 3DFin==0.4.1 as python plugin (with qt5 support not latest version)
python -m pip install -r ${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt

eval $(
    source /etc/lsb-release;
    echo DISTRIB_ID="$DISTRIB_ID";
    echo DISTRIB_RELEASE="$DISTRIB_RELEASE"
)

set -x # Echo commands on
# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

# Build options - separate common options for clarity
BUILD_OPTIONS="with_pcl_nurbs package_installer plugin_treeiso"

# Add with_rdb option based on WITH_RDB environment variable
# Default: OFF (for Dockerfile_build), ON (for Dockerfile.ci and Dockerfile.ci.qt6)
if [ -z "${WITH_RDB:-}" ]; then
    # If not set, default to OFF (for backward compatibility with Dockerfile_build)
    WITH_RDB=OFF
fi

if [ "${WITH_RDB}" = "ON" ]; then
    BUILD_OPTIONS="${BUILD_OPTIONS} with_rdb"
    echo "PLUGIN_IO_QRDB will be enabled"
else
    BUILD_OPTIONS="${BUILD_OPTIONS} without_rdb"
    echo "PLUGIN_IO_QRDB will be disabled"
fi

if [ "${ONLY_BUILD_CUDA}" = "ON" ]; then
    echo "Start to build GUI package with CUDA..."
    echo
    export BUILD_CUDA_MODULE=ON
    build_gui_app ${BUILD_OPTIONS}
    echo
else
    echo "Start to build GUI package with only CPU..."
    echo
    export BUILD_CUDA_MODULE=OFF
    build_gui_app ${BUILD_OPTIONS}
    echo

    # Building with cuda if cuda available
    if [ "${BUILD_CUDA_MODULE_FLAG}" = "ON" ]; then
        echo "Start to build GUI package with CUDA..."
        echo
        export BUILD_CUDA_MODULE=ON
        build_gui_app ${BUILD_OPTIONS}
        echo
    fi
fi

echo "Finish building ACloudViewer GUI to $ACloudViewer_INSTALL"

set +x # Echo commands off
echo
