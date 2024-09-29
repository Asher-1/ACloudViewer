#!/bin/bash
set -euo pipefail

export BUILD_CUDA_MODULE=ON
export BUILD_SHARED_LIBS=OFF
export DEVELOPER_BUILD=OFF
export BUILD_PYTORCH_OPS=OFF
export BUILD_TENSORFLOW_OPS=OFF

set -x # Echo commands on
# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${ACloudViewer_DEV}/ACloudViewer/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

echo "Start to build GUI package..."
echo
build_gui_app package_installer
echo

echo "Finish building ACloudViewer GUI to $ACloudViewer_INSTALL"

set +x # Echo commands off
echo
