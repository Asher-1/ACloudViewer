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
export NPROC=$(nproc)
export POST_FIX="python${PYTHON_VERSION}"
export ENV_NAME="cloudViewer"
echo "ENV_NAME: " ${ENV_NAME}

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

set +u
if [ -n "$CONDA_EXE" ]; then
    CONDA_ROOT=$(dirname $(dirname "$CONDA_EXE"))
elif [ -n "$CONDA_PREFIX" ]; then
    CONDA_ROOT=$(dirname "$CONDA_PREFIX")
else
    echo "Failed to find Miniconda3 install path..."
    exit -1
fi
set -u

echo "source $CONDA_ROOT/etc/profile.d/conda.sh"
source "$CONDA_ROOT/etc/profile.d/conda.sh"

conda config --set always_yes yes
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "env $ENV_NAME exists and start to remove..."
    conda env remove -n $ENV_NAME
fi

echo "conda env create..."
export CONDA_PREFIX="$CONDA_ROOT/envs/${ENV_NAME}"
conda create -y -n ${ENV_NAME} python=${PYTHON_VERSION} \
 && conda activate ${ENV_NAME} \
 && which python \
 && python --version

# for python plugin
python -m pip install -r ${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt

eval $(
    source /etc/lsb-release;
    echo DISTRIB_ID="$DISTRIB_ID";
    echo DISTRIB_RELEASE="$DISTRIB_RELEASE"
)

# if [ "$DISTRIB_ID" == "Ubuntu" ] && [ "$DISTRIB_RELEASE" != "18.04" ]; then
#     # fix the library conflicts between ubuntu and conda  about incorrect link issues from ibffi.so.7 to libffi.so.8.1.0
#     if [ "${PYTHON_VERSION}" = "3.8" ]; then
#         echo -e "\ny" | conda install libffi==3.3
#     else
#         echo -e "\ny" | conda install libffi==3.4.2
#     fi
# fi

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
