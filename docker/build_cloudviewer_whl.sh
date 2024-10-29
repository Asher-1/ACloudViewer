#!/bin/bash
set -euo pipefail

export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_CUDA_MODULE=ON
export BUILD_PYTORCH_OPS=ON
export BUILD_TENSORFLOW_OPS=OFF
export PYTHON_VERSION=$1
export NPROC=$(nproc)
export ENV_NAME="python${PYTHON_VERSION}"
echo "ENV_NAME: " ${ENV_NAME}

echo "conda activate..."
export CONDA_PREFIX="/root/miniconda3/envs/${ENV_NAME}"
export PATH="/root/miniconda3/envs/${ENV_NAME}/bin:${PATH}"
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION} \
 && conda activate ${ENV_NAME} \
 && which python \
 && python --version

 # fix the library conflicts between ubuntu2204 and conda  about incorrect link issues from ibffi.so.7 to libffi.so.8.1.0
# echo -e "\ny" | conda install libffi==3.3

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${ACloudViewer_DEV}/ACloudViewer/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"
install_python_dependencies speedup with-cuda with-jupyter with-unit-test
build_pip_package build_realsense build_azure_kinect build_jupyter
# build_pip_package build_azure_kinect build_jupyter

set -x # Echo commands on
df -h
# Run on GPU only. CPU versions run on Github already
if nvidia-smi >/dev/null 2>&1; then
    echo "Try importing cloudViewer Python package"
    test_wheel ${ACloudViewer_BUILD}/lib/python_package/pip_package/cloudViewer-*whl
    df -h
    # echo "Running cloudViewer Python tests..."
    # run_python_tests
    # echo
fi

echo "Finish building cloudViewer wheel based on ${PYTHON_VERSION}!"
echo "mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}"
mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}

echo "Backup whl package to ${ACloudViewer_INSTALL}"
set +x # Echo commands off
echo
