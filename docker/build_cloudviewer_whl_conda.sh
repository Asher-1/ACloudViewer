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

if [ -n "$CONDA_EXE" ]; then
    CONDA_ROOT=$(dirname $(dirname "$CONDA_EXE"))
elif [ -n "$CONDA_PREFIX" ]; then
    CONDA_ROOT=$(dirname "$CONDA_PREFIX")
else
    echo "Failed to find Miniconda3 install path..."
    exit -1
fi

if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "env $ENV_NAME exists and start to remove..."
    conda env remove -n $ENV_NAME
fi

echo "conda activate..."
export CONDA_PREFIX="/root/miniconda3/envs/${ENV_NAME}"
export PATH="/root/miniconda3/envs/${ENV_NAME}/bin:${PATH}"
cp ${ACloudViewer_DEV}/ACloudViewer/.ci/conda_env.yml /root/conda_env_${ENV_NAME}.yml
sed -i "s/3.8/${PYTHON_VERSION}/g" /root/conda_env_${ENV_NAME}.yml
conda env create -f /root/conda_env_${ENV_NAME}.yml
# conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION} \
conda activate ${ENV_NAME} \
 && which python \
 && python --version

if [ $? -eq 0 ]; then
    echo "env $ENV_NAME activate successfully"
    echo "current Python path: $(which python)"
    echo "current Python version: $(python --version)"
else
    echo "Activate failed, please run mannually: conda activate $ENV_NAME"
fi

 # fix the library conflicts between ubuntu2204 and conda  about incorrect link issues from ibffi.so.7 to libffi.so.8.1.0
# echo -e "\ny" | conda install libffi==3.3
# [mv relocation issues] fix the issues about the conflicts with libattr between conda and system
if [ -f "$CONDA_PREFIX/lib/libattr.so.1" ]; then
    echo "fix issues with system about: $CONDA_PREFIX/lib/libattr.so.1"
    mv $CONDA_PREFIX/lib/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.2
    ln -s /lib/x86_64-linux-gnu/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${ACloudViewer_DEV}/ACloudViewer/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"
install_python_dependencies with-cuda with-jupyter with-unit-test
# build_pip_package with_conda build_realsense build_azure_kinect build_jupyter
build_pip_package with_conda build_azure_kinect build_jupyter

set -x # Echo commands on
df -h
# Run on GPU only. CPU versions run on Github already
if nvidia-smi >/dev/null 2>&1; then
    echo "Try importing cloudViewer Python package"
    test_wheel ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl
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
