#!/bin/bash
set -euo pipefail

export BUILD_CUDA_MODULE=ON
export BUILD_SHARED_LIBS=OFF
export DEVELOPER_BUILD=OFF
export BUILD_PYTORCH_OPS=OFF
export BUILD_TENSORFLOW_OPS=OFF
export PYTHON_VERSION=$1
export NPROC=$(nproc)
export ENV_NAME="cloudViewer"
echo "ENV_NAME: " ${ENV_NAME}

echo "conda activate..."
export PATH="/root/miniconda3/envs/${ENV_NAME}/bin:${PATH}"
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION} \
 && conda activate ${ENV_NAME} \
 && which python \
 && python --version \
 && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn \
 && pip config list

 # fix the library conflicts between ubuntu2204 and conda  about incorrect link issues from ibffi.so.7 to libffi.so.8.1.0
echo -e "\ny" | conda install libffi==3.3

set -x # Echo commands on
# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${ACloudViewer_DEV}/ACloudViewer/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

echo "Start to build GUI package..."
echo
build_gui_app with_pcl_nurbs with_gdal package_installer
echo

echo "Finish building ACloudViewer GUI to $ACloudViewer_INSTALL"

set +x # Echo commands off
echo
