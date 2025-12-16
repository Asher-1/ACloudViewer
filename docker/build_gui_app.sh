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

if [ "${ONLY_BUILD_CUDA}" = "ON" ]; then
    echo "Start to build GUI package with CUDA..."
    echo
    export BUILD_CUDA_MODULE=ON
    build_gui_app with_pcl_nurbs package_installer plugin_treeiso
    echo
else
    echo "Start to build GUI package with only CPU..."
    echo
    export BUILD_CUDA_MODULE=OFF
    build_gui_app with_pcl_nurbs package_installer plugin_treeiso
    echo

    # Building with cuda if cuda available
    if [ "${BUILD_CUDA_MODULE_FLAG}" = "ON" ]; then
        echo "Start to build GUI package with CUDA..."
        echo
        export BUILD_CUDA_MODULE=ON
        build_gui_app with_pcl_nurbs package_installer plugin_treeiso
        echo
    fi
fi

echo "Finish building ACloudViewer GUI to $ACloudViewer_INSTALL"

set +x # Echo commands off
echo
