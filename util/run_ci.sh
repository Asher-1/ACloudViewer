#!/usr/bin/env bash
set -euo pipefail

# this scripts only used on macos
set -x # Echo commands on
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

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
conda env create -f ${CLOUDVIEWER_SOURCE_ROOT}/.ci/conda_macos_cloudViewer.yml
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
set -x # Echo commands on

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source "$(dirname "$0")"/ci_utils.sh

echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

echo "Start to build GUI package On MacOS..."
echo
build_gui_app with_conda package_installer


df -h