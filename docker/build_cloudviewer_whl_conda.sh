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
export ENV_NAME="python${PYTHON_VERSION}"
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
cp ${CLOUDVIEWER_SOURCE_ROOT}/.ci/conda_linux.yml /root/conda_env_${ENV_NAME}.yml
sed -i "s/3.8/${PYTHON_VERSION}/g" /root/conda_env_${ENV_NAME}.yml
conda env create -f /root/conda_env_${ENV_NAME}.yml
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

# [mv relocation issues] fix the issues about the conflicts with libattr between conda and system
if [ -f "$CONDA_PREFIX/lib/libattr.so.1" ]; then
    echo "fix issues with system about: $CONDA_PREFIX/lib/libattr.so.1"
    mv $CONDA_PREFIX/lib/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.2
    ln -s /lib/x86_64-linux-gnu/libattr.so.1 $CONDA_PREFIX/lib/libattr.so.1
fi

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake"
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"
install_python_dependencies with-jupyter with-unit-test
build_pip_package with_conda build_realsense build_azure_kinect build_jupyter
# build_pip_package with_conda build_azure_kinect build_jupyter

set -x # Echo commands on
df -h

eval $(
    source /etc/lsb-release;
    echo DISTRIB_ID="$DISTRIB_ID";
    echo DISTRIB_RELEASE="$DISTRIB_RELEASE"
)
if [ "$DISTRIB_ID" == "Ubuntu" -a "$DISTRIB_RELEASE" == "22.04" ]; then
    # fix GLIB_*_30 missing issues
    echo "due to GLIB_ missing issues on Ubuntu22.04 and ignore test"
elif [ "$IGNORE_TEST" == "ON" ]; then
    echo "Ignore test in internal docker when github action running!"
else
    pushd build # PWD=ACloudViewer/build
    echo "Try importing cloudviewer Python package"
    if [ "${BUILD_CUDA_MODULE}" = "ON" ]; then
        test_wheel ${ACloudViewer_BUILD}/lib/python_package/pip_package/cloudviewer-*whl
    else
        test_wheel ${ACloudViewer_BUILD}/lib/python_package/pip_package/cloudviewer_cpu-*whl
    fi
    popd # PWD=ACloudViewer
fi

echo "Finish building cloudviewer wheel based on ${PYTHON_VERSION}!"
echo "mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}"
mv ${ACloudViewer_BUILD}/lib/python_package/pip_package/*whl ${ACloudViewer_INSTALL}

echo "Backup whl package to ${ACloudViewer_INSTALL}"
set +x # Echo commands off
echo
