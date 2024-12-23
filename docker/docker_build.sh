#!/usr/bin/env bash
#
# docker_build.sh is used to build cloudViewer docker images for all supported
# scenarios. This can be used in CI and on local machines. The objective is to
# allow developers to emulate CI environments for debugging or build release
# artifacts such as Python wheels locally.
#
# Guidelines:
# - Use a flat list of options.
#   We don't want to have a cartesian product of different combinations of
#   options. E.g., to support Ubuntu {20.04, 24.04} with Python {3.8, 3.9}, we
#   don't specify the OS and Python version separately, instead, we have a flat
#   list of combinations: [u2004_py39, u2004_py310, u2404_py39, u2404_py310].
# - No external environment variables.
#   This script should not make assumptions on external environment variables.
#   This make the Docker image reproducible across different machines.
set -euo pipefail

export BUILDKIT_PROGRESS=plain

__usage_docker_build="USAGE:
    $(basename $0) [OPTION]

OPTION:

    # Ubuntu CPU CI (Dockerfile.ci)
    cpu-static                  : Ubuntu CPU static

    # ML CIs (Dockerfile.ci)
    2-focal                   : CUDA CI, 2-bionic, developer mode
    5-ml-jammy                 : CUDA CI, 5-ml-focal, developer mode

    # CUDA wheels (Dockerfile.ci)
    cuda_wheel_py38_dev        : CUDA Python 3.8 wheel, developer mode
    cuda_wheel_py39_dev        : CUDA Python 3.9 wheel, developer mode
    cuda_wheel_py310_dev       : CUDA Python 3.10 wheel, developer mode
    cuda_wheel_py311_dev       : CUDA Python 3.11 wheel, developer mode
    cuda_wheel_py38            : CUDA Python 3.8 wheel, release mode
    cuda_wheel_py39            : CUDA Python 3.9 wheel, release mode
    cuda_wheel_py310           : CUDA Python 3.10 wheel, release mode
    cuda_wheel_py311           : CUDA Python 3.11 wheel, release mode
    cuda_wheel_py312           : CUDA Python 3.12 wheel, release mode
"

HOST_CLOUDVIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# Shared variables
CCACHE_VERSION=4.3
CMAKE_VERSION=cmake-3.29.2-linux-x86_64
CMAKE_VERSION_AARCH64=cmake-3.24.4-linux-aarch64
CUDA_VERSION=11.8.0-cudnn8
CUDA_VERSION_LATEST=11.8.0-cudnn8
UBUNTU_FOCAL=20.04
UBUNTU_JAMMY=22.04
UBUNTU_VERSION=$UBUNTU_FOCAL

print_usage_and_exit_docker_build() {
    echo "$__usage_docker_build"
    exit 1
}

cuda_wheel_build() {
    BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
    CCACHE_TAR_NAME=cloudViewer-ubuntu-2004-cuda-ci-ccache
    export BUILD_SHARED_LIBS=OFF

    if [ "${UBUNTU_VERSION}" = "${UBUNTU_JAMMY}" ]; then
        QT_BASE_DIR="/usr/lib/x86_64-linux-gnu/qt5"
    else 
        QT_BASE_DIR="/opt/qt515"
    fi

    options="$(echo "$@" | tr ' ' '|')"
    echo "[cuda_wheel_build()] options: ${options}"
    if [[ "py38" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.8
    elif [[ "py39" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.9
    elif [[ "py310" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.10
    elif [[ "py311" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.11
    elif [[ "py312" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.12
    else
        echo "Invalid python version."
        print_usage_and_exit_docker_build
    fi
    if [[ "dev" =~ ^($options)$ ]]; then
        DEVELOPER_BUILD=ON
    else
        DEVELOPER_BUILD=OFF
    fi
    echo "[cuda_wheel_build()] PYTHON_VERSION: ${PYTHON_VERSION}"
    echo "[cuda_wheel_build()] QT_BASE_DIR: ${QT_BASE_DIR}"
    echo "[cuda_wheel_build()] UBUNTU_VERSION: ${UBUNTU_VERSION}"
    echo "[cuda_wheel_build()] DEVELOPER_BUILD: ${DEVELOPER_BUILD}"
    echo "[cuda_wheel_build()] BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}"
    echo "[cuda_wheel_build()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[cuda_wheel_build()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:?'env var must be set.'}"
    echo "[cuda_wheel_build()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:?'env var must be set.'}"

    pushd "${HOST_CLOUDVIEWER_ROOT}"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg QT_BASE_DIR="${QT_BASE_DIR}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        --build-arg CCACHE_TAR_NAME="${CCACHE_TAR_NAME}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg CCACHE_VERSION="${CCACHE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}" \
        --build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        --build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        --build-arg BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        --build-arg PACKAGE="OFF" \
        --build-arg CI="${CI:-}" \
        -t cloudviewer-ci:wheel \
        -f docker/Dockerfile.wheel .
    popd

    docker run -v "${PWD}:/opt/mount" --rm cloudviewer-ci:wheel \
        bash -c "cp /root/install/cloudViewer*.whl /opt/mount \
            && chown $(id -u):$(id -g) /opt/mount/cloudViewer*.whl"
}

ci_build() {
    echo "[ci_build()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[ci_build()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[ci_build()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[ci_build()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
    echo "[ci_build()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[ci_build()] CCACHE_VERSION=${CCACHE_VERSION}"
    echo "[ci_build()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[ci_build()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[ci_build()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[ci_build()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[ci_build()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"

    pushd "${HOST_CLOUDVIEWER_ROOT}"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        --build-arg CCACHE_TAR_NAME="${CCACHE_TAR_NAME}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg CCACHE_VERSION="${CCACHE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        --build-arg BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}" \
        --build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        --build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        --build-arg PACKAGE="ON" \
        --build-arg CI="${CI:-}" \
        -t "${DOCKER_TAG}" \
        -f docker/Dockerfile.ci .
    popd

    docker run -v "${PWD}:/opt/mount" --rm "${DOCKER_TAG}" \
        bash -cx "cp /root/install/*run /opt/mount \
               && chown $(id -u):$(id -g) /opt/mount/*run"
}

cuda-focal_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-focal

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_FOCAL}
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=cloudviewer-ci-cuda-focal
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-jammy_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-jammy

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_JAMMY}
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=cloudviewer-ci-cuda-jammy
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-focal-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-focal

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_FOCAL}
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=cloudviewer-ci-cuda-focal
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-jammy-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-jammy

    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_JAMMY}
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=cloudviewer-ci-cuda-jammy
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-focal_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-focal

    export BASE_IMAGE=ubuntu:${UBUNTU_FOCAL}
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=cloudviewer-ci-cpu-focal
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-jammy_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-jammy
    export BASE_IMAGE=ubuntu:${UBUNTU_JAMMY}
    export DEVELOPER_BUILD=ON
    export CCACHE_TAR_NAME=cloudviewer-ci-cpu-jammy
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-focal-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-focal

    export BASE_IMAGE=ubuntu:${UBUNTU_FOCAL}
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=cloudviewer-ci-cpu-focal
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-jammy-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-jammy
    export BASE_IMAGE=ubuntu:${UBUNTU_JAMMY}
    export DEVELOPER_BUILD=OFF
    export CCACHE_TAR_NAME=cloudviewer-ci-cpu-jammy
    export PYTHON_VERSION=3.11
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

function main() {
    if [[ "$#" -ne 1 ]]; then
        echo "Error: invalid number of arguments: $#." >&2
        print_usage_and_exit_docker_build
    fi
    echo "[$(basename $0)] building $1"
    case "$1" in

    # CPU CI
    cpu-focal)
        cpu-focal_export_env
        ci_build
        ;;
    cpu-jammy)
        cpu-jammy_export_env
        ci_build
        ;;
    cpu-focal-release)
        cpu-focal-release_export_env
        ci_build
        ;;
    cpu-jammy-release)
        cpu-jammy-release_export_env
        ci_build
        ;;

    # CUDA CIs
    cuda-focal)
        cuda-focal_export_env
        ci_build
        ;;
    cuda-jammy)
        cuda-jammy_export_env
        ci_build
        ;;
    cuda-focal-release)
        cuda-focal-release_export_env
        ci_build
        ;;
    cuda-jammy-release)
        cuda-jammy-release_export_env
        ci_build
        ;;

    # CUDA wheels
    cuda_wheel_py38_dev)
        cuda_wheel_build py38 dev
        ;;
    cuda_wheel_py39_dev)
        cuda_wheel_build py39 dev
        ;;
    cuda_wheel_py310_dev)
        cuda_wheel_build py310 dev
        ;;
    cuda_wheel_py311_dev)
        cuda_wheel_build py311 dev
        ;;
    cuda_wheel_py312_dev)
        cuda_wheel_build py312 dev
        ;;
    cuda_wheel_py38)
        cuda_wheel_build py38
        ;;
    cuda_wheel_py39)
        cuda_wheel_build py39
        ;;
    cuda_wheel_py310)
        cuda_wheel_build py310
        ;;
    cuda_wheel_py311)
        cuda_wheel_build py311
        ;;
    cuda_wheel_py312)
        cuda_wheel_build py312
        ;;
    *)
        echo "Error: invalid argument: ${1}." >&2
        print_usage_and_exit_docker_build
        ;;
    esac
}

# main() will be executed when ./docker_build.sh is called directly.
# main() will not be executed when ./docker_build.sh is sourced.
if [ "$0" = "$BASH_SOURCE" ]; then
    main "$@"
fi
