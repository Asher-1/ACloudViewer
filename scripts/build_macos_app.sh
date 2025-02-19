#!/usr/bin/env bash
set -euo pipefail

# Requirements as follows:
# conda env create -f .ci/conda_macos_cloudViewer.yml
# conda activate python3.8

# use lower target(11.0) version for compacibility
export MACOSX_DEPLOYMENT_TARGET=11.0
export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_CUDA_MODULE=OFF
export BUILD_PYTORCH_OPS=OFF
export BUILD_TENSORFLOW_OPS=OFF
export PYTHON_VERSION=$1
export ACloudViewer_INSTALL=~/cloudViewer_install
export ENV_NAME="cloudViewer"
export NPROC=$(nproc)
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

set -x # Echo commands on
if [ -n "$CONDA_EXE" ]; then
    CONDA_ROOT=$(dirname $(dirname "$CONDA_EXE"))
elif [ -n "$CONDA_PREFIX" ]; then
    CONDA_ROOT=$(dirname "$CONDA_PREFIX")
else
    echo "Failed to find Miniconda3 install path..."
    exit -1
fi

echo "source $CONDA_ROOT/etc/profile.d/conda.sh"
source "$CONDA_ROOT/etc/profile.d/conda.sh"

conda config --set always_yes yes
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "env $ENV_NAME exists and start to remove..."
    conda env remove -n $ENV_NAME
fi

echo "conda env create and activate..."
export CONDA_PREFIX="${CONDA_ROOT}/envs/${ENV_NAME}"
cp ${CLOUDVIEWER_SOURCE_ROOT}/.ci/conda_macos_cloudViewer.yml /tmp/conda_macos_cloudViewer.yml
sed -i "" "s/3.8/${PYTHON_VERSION}/g" /tmp/conda_macos_cloudViewer.yml
conda env create -f /tmp/conda_macos_cloudViewer.yml
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

if [ -z "$CONDA_PREFIX" ] ; then
    echo "Conda env is not activated"
else
    echo "Conda env now is $CONDA_PREFIX"
fi

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH
set +x # Echo commands off

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh

echo "Start to build GUI package On MacOS..."
echo
build_gui_app with_conda package_installer
echo "Install ACloudViewer package to ${ACloudViewer_INSTALL}"
echo
