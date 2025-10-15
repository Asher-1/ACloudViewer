#!/bin/bash
set -euo pipefail

PACKAGE=${PACKAGE:-OFF}
IGNORE_TEST=${IGNORE_TEST:-OFF}
DEVELOPER_BUILD=${DEVELOPER_BUILD:-OFF}
BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-OFF}
BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE:-ON}
BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-ON}
BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-OFF}

export PACKAGE=${PACKAGE}
export IGNORE_TEST=${IGNORE_TEST}
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

set -x # Echo commands on
df -h

eval $(
    source /etc/lsb-release;
    echo DISTRIB_ID="$DISTRIB_ID";
    echo DISTRIB_RELEASE="$DISTRIB_RELEASE"
)
if [ "$IGNORE_TEST" == "ON" ]; then
    echo "Ignore test in internal docker when github action running!"
else
    pushd build # PWD=ACloudViewer/build
    echo "Try importing cloudViewer Python package"
    if [ "${BUILD_CUDA_MODULE}" = "ON" ]; then
        test_wheel ${ACloudViewer_BUILD}/lib/python_package/pip_package/cloudViewer-*whl
    else
        test_wheel ${ACloudViewer_BUILD}/lib/python_package/pip_package/cloudViewer_cpu-*whl
    fi
    popd # PWD=ACloudViewer
fi

echo "Finish building cloudViewer wheel based on ${PYTHON_VERSION}!"
echo "mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}"
mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}

echo "Backup whl package to ${ACloudViewer_INSTALL}"
set +x # Echo commands off
echo
