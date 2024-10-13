#!/usr/bin/env bash
set -euo pipefail

# conda env create -f /root/conda_macos.yml
# conda activate python3.8

export DEVELOPER_BUILD=OFF
export BUILD_SHARED_LIBS=OFF
export BUILD_CUDA_MODULE=OFF
export BUILD_PYTORCH_OPS=OFF
export BUILD_TENSORFLOW_OPS=OFF

CLOUDVIEWER_SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# this scripts only used on macos
if [ -z "$CONDA_PREFIX" ] ; then
	echo "Conda env is not activated"
	exit -1
fi

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source ${CLOUDVIEWER_SOURCE_ROOT}/util/ci_utils.sh

echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

install_python_dependencies with-unit-test purge-cache

build_all