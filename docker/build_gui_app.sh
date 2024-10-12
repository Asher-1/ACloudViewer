#!/bin/bash
set -euo pipefail

export BUILD_CUDA_MODULE=ON
export BUILD_SHARED_LIBS=OFF
export DEVELOPER_BUILD=OFF
export BUILD_PYTORCH_OPS=OFF
export BUILD_TENSORFLOW_OPS=OFF
export PYTHON_VERSION=$1
export NPROC=$(nproc)
export ENV_NAME="python${PYTHON_VERSION}"
echo "ENV_NAME: " ${ENV_NAME}

echo "conda activate..."
export CONDA_PREFIX="/root/miniconda3/envs/cloudViewer"
export PATH="/root/miniconda3/envs/cloudViewer/bin:${PATH}"
cp ${ACloudViewer_DEV}/ACloudViewer/.ci/conda_cloudViewer.yml /root/conda_cloudViewer_${ENV_NAME}.yml
sed -i "s/3.8/${PYTHON_VERSION}/g" /root/conda_cloudViewer_${ENV_NAME}.yml
conda env create -f /root/conda_cloudViewer_${ENV_NAME}.yml
conda activate cloudViewer \
 && which python \
 && python --version \
 && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn \
 && pip config list

# [mv relocation issues] fix the issues about the conflicts with libattr between conda and system
if [ -f "$CONDA_PREFIX/lib/libattr.so.1" ]; then
    echo "fix issues with system about: $CONDA_PREFIX/lib/libattr.so.1"
    mv $CONDA_PREFIX/lib/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.2
    ln -s /lib/x86_64-linux-gnu/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.1
fi
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

set -x # Echo commands on
# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${ACloudViewer_DEV}/ACloudViewer/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

echo "Start to build GUI package..."
echo
build_gui_app package_installer
# not conda env build condition
# build_gui_app with_pcl_nurbs with_gdal package_installer
echo

echo "Finish building ACloudViewer GUI to $ACloudViewer_INSTALL"

set +x # Echo commands off
echo
