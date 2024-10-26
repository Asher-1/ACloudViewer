#!/usr/bin/env bash
set -euo pipefail

# Requirements as follows:
# conda env create -f .ci/conda_macos.yml
# conda activate python3.8

# From build/bin directory:
# ./scripts/platforms/mac/sign_macos_app.sh ~/cloudViewer_install/ACloudViewer/ACloudViewer.app ./eCV/Mac/ACloudViewer.entitlements <apple-id> <cert-name> <team-id> <app-password>

# test wheel
# python -W default -c "import cloudViewer; print('Installed:', cloudViewer); print('BUILD_CUDA_MODULE: ', cloudViewer._build_config['BUILD_CUDA_MODULE'])"
# python -W default -c "import cloudViewer; print('CUDA available: ', cloudViewer.core.cuda.is_available())"
# python -W default -c "import cloudViewer.ml.torch; print('PyTorch Ops library loaded:', cloudViewer.ml.torch._loaded)"

export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_CUDA_MODULE=OFF
export BUILD_PYTORCH_OPS=OFF
export BUILD_TENSORFLOW_OPS=OFF
export ACloudViewer_INSTALL=~/cloudViewer_install
export NPROC=$(nproc)
# export NPROC=$(($(nproc) * 2))
echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"
CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# if [[ ! -f "$ACloudViewer_INSTALL/ACloudViewer*.dmg" ]]; then
if ! find "$ACloudViewer_INSTALL" -maxdepth 1 -name "ACloudViewer*.dmg" | grep -q .; then
    ENV_NAME="cloudViewer"
    set -x # Echo commands on
    if [ -n "$CONDA_EXE" ]; then
        CONDA_ROOT=$(dirname $(dirname "$CONDA_EXE"))
    elif [ -n "$CONDA_PREFIX" ]; then
        CONDA_ROOT=$(dirname "$CONDA_PREFIX")
    else
        echo "Failed to find Miniconda3 install path..."
        exit -1
    fi

    source "$CONDA_ROOT/etc/profile.d/conda.sh"

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

    export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$CONDA_PREFIX/lib:$PKG_CONFIG_PATH
    export PATH=$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$CONDA_PREFIX/lib:$PATH

    if [ -z "$CONDA_PREFIX" ] ; then
        echo "Conda env is not activated"
    else
        echo "Conda env now is $CONDA_PREFIX"
    fi
    set +x # Echo commands off

    # Get build scripts and control environment variables
    # shellcheck source=ci_utils.sh
    source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh

    echo "Start to build GUI package On MacOS..."
    echo
    build_gui_app with_conda package_installer
    echo "Install ACloudViewer package to ${ACloudViewer_INSTALL}"
    echo
else
    echo "Ignore GUI app building due to have builded before..."
fi
echo

echo "Start to build wheel for python3.8-3.11 On MacOS..."
echo
CLOUDVIEWER_BUILD_DIR=${CLOUDVIEWER_SOURCE_ROOT}/build
MACOS_WHL_BUILD_SHELL=${CLOUDVIEWER_SOURCE_ROOT}/scripts/build_macos_whl.sh
rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.8
rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.9
rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.10
rm -rf ${CLOUDVIEWER_BUILD_DIR}/* && ${MACOS_WHL_BUILD_SHELL} 3.11
echo "All install to ${ACloudViewer_INSTALL}"
echo