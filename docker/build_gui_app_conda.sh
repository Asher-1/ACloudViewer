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
cp ${CLOUDVIEWER_SOURCE_ROOT}/.ci/conda_cloudViewer.yml /root/conda_cloudViewer_${POST_FIX}.yml
sed -i "s/3.8/${PYTHON_VERSION}/g" /root/conda_cloudViewer_${POST_FIX}.yml
conda env create -f /root/conda_cloudViewer_${POST_FIX}.yml
conda activate $ENV_NAME \
 && which python \
 && python --version \
 && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
 && pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn \
 && pip config list

 # for python plugin
python -m pip install -r ${CLOUDVIEWER_SOURCE_ROOT}/plugins/core/Standard/qPythonRuntime/requirements-release.txt

# [mv relocation issues] fix the issues about the conflicts with libattr between conda and system
if [ -f "$CONDA_PREFIX/lib/libattr.so.1" ]; then
    echo "fix issues with system about: $CONDA_PREFIX/lib/libattr.so.1"
    mv $CONDA_PREFIX/lib/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.2
    ln -s /lib/x86_64-linux-gnu/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake"
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
export GMP_DIR=$CONDA_PREFIX
export MPFR_DIR=$CONDA_PREFIX

# fix no such file: /usr/lib/libGL.so when building libPCLEngine
if [ -f "/usr/lib/libGL.so" ]; then
    echo "/usr/lib/libGL.so has been already, no need to soft link!"
else
    echo "fix issues with no file in /usr/lib/libGL.so"
    ln -s /usr/lib/x86_64-linux-gnu/libGL.so /usr/lib/libGL.so
fi
# fix libQt undefined issues
if [ -f "/usr/lib/x86_64-linux-gnu/libstdc++.so.6" ]; then
    echo "ln -s $CONDA_PREFIX/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    rm /usr/lib/x86_64-linux-gnu/libstdc++.so.6
    ln -s $CONDA_PREFIX/lib/libstdc++.so.6 /usr/lib/x86_64-linux-gnu/libstdc++.so.6
fi

set -x # Echo commands on
# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

if [ "${ONLY_BUILD_CUDA}" = "ON" ]; then
    echo "Start to build GUI package with CUDA..."
    echo
    export BUILD_CUDA_MODULE=ON
    build_gui_app with_conda package_installer plugin_treeiso
    echo
else
    echo "Start to build GUI package with only CPU..."
    echo
    export BUILD_CUDA_MODULE=OFF
    build_gui_app with_conda package_installer plugin_treeiso
    echo

    # Building with cuda if cuda available
    if [ "${BUILD_CUDA_MODULE_FLAG}" = "ON" ]; then
        echo "Start to build GUI package with CUDA..."
        echo
        export BUILD_CUDA_MODULE=ON
        build_gui_app with_conda package_installer plugin_treeiso
        echo
    fi
fi

echo "Finish building ACloudViewer GUI to $ACloudViewer_INSTALL"

set +x # Echo commands off
echo
