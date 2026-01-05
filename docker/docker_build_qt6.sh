#!/usr/bin/env bash
#
# docker_build_qt6.sh is used to build cloudViewer docker images with Qt6 support.
# This builds with PCL 1.15.1, VTK 9.4.2, and Qt 6.xxx.
#
# Usage:
#   ./docker_build_qt6.sh [OPTION]
#
# Options are similar to docker_build.sh but use Qt6-based Dockerfile.
#
set -euo pipefail

export BUILDKIT_PROGRESS=plain

__usage_docker_build_qt6="USAGE:
    $(basename $0) [OPTION]

OPTION:

    # Ubuntu CPU CI with Qt6 (Dockerfile.ci.qt6)
    cpu-focal-qt6             : Ubuntu 20.04 CPU with Qt6 (via aqtinstall), developer mode
    cpu-jammy-qt6             : Ubuntu 22.04 CPU with Qt6, developer mode
    cpu-noble-qt6             : Ubuntu 24.04 CPU with Qt6, developer mode
    cpu-focal-qt6-release     : Ubuntu 20.04 CPU with Qt6 (via aqtinstall), release mode
    cpu-jammy-qt6-release     : Ubuntu 22.04 CPU with Qt6, release mode
    cpu-noble-qt6-release     : Ubuntu 24.04 CPU with Qt6, release mode

    # CUDA CI with Qt6 (Dockerfile.ci.qt6)
    cuda-focal-qt6            : CUDA Ubuntu 20.04 with Qt6 (via aqtinstall), developer mode
    cuda-jammy-qt6            : CUDA Ubuntu 22.04 with Qt6, developer mode
    cuda-noble-qt6            : CUDA Ubuntu 24.04 with Qt6, developer mode
    cuda-focal-qt6-release    : CUDA Ubuntu 20.04 with Qt6 (via aqtinstall), release mode
    cuda-jammy-qt6-release    : CUDA Ubuntu 22.04 with Qt6, release mode
    cuda-noble-qt6-release    : CUDA Ubuntu 24.04 with Qt6, release mode

Build dependencies:
    - VTK 9.4.2
    - PCL 1.15.1
    - Qt 6.x (system packages for jammy/noble, aqtinstall for focal)
"

HOST_CLOUDVIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# Shared variables
CMAKE_VERSION=cmake-3.29.2-linux-x86_64
CUDA_VERSION=12.6.3-cudnn
CUDA_VERSION_LATEST=12.6.3-cudnn
UBUNTU_FOCAL=20.04
UBUNTU_JAMMY=22.04
UBUNTU_NOBLE=24.04

print_usage_and_exit_docker_build_qt6() {
    echo "$__usage_docker_build_qt6"
    exit 1
}

ci_build_qt6() {
    # For Qt6, use system Qt6 path
    QT_BASE_DIR="/usr/lib/x86_64-linux-gnu/qt6"

    echo "[ci_build_qt6()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[ci_build_qt6()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[ci_build_qt6()] QT_BASE_DIR: ${QT_BASE_DIR}"
    echo "[ci_build_qt6()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[ci_build_qt6()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[ci_build_qt6()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[ci_build_qt6()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[ci_build_qt6()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[ci_build_qt6()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[ci_build_qt6()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[ci_build_qt6()] Using VTK 9.4.2, PCL 1.15.1, Qt6"

    pushd "${HOST_CLOUDVIEWER_ROOT}"
    docker build \
        --network host \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg QT_BASE_DIR="${QT_BASE_DIR}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        --build-arg BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}" \
        --build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        --build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        --build-arg PACKAGE="ON" \
        --build-arg CI="${CI:-}" \
        -t "${DOCKER_TAG}" \
        -f docker/Dockerfile.ci.qt6 .
    popd

    docker run -v "${PWD}:/opt/mount" --rm "${DOCKER_TAG}" \
        bash -cx "cp /root/install/*run /opt/mount \
               && chown $(id -u):$(id -g) /opt/mount/*run"
}

# CPU builds with Qt6
cpu-focal-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-focal-qt6

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.10
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-jammy-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-jammy-qt6

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-noble-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-noble-qt6

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-focal-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-focal-qt6

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.10
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-jammy-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-jammy-qt6

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-noble-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-noble-qt6

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

# CUDA builds with Qt6
cuda-focal-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-focal-qt6

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.10
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-jammy-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-jammy-qt6

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-noble-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-noble-qt6

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-focal-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-focal-qt6

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.10
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-jammy-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-jammy-qt6

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-noble-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-noble-qt6

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

function main() {
    if [[ "$#" -ne 1 ]]; then
        echo "Error: invalid number of arguments: $#." >&2
        print_usage_and_exit_docker_build_qt6
    fi
    echo "[$(basename $0)] building $1 with Qt6 support"
    case "$1" in

    # CPU CI with Qt6
    cpu-focal-qt6)
        cpu-focal-qt6_export_env
        ci_build_qt6
        ;;
    cpu-jammy-qt6)
        cpu-jammy-qt6_export_env
        ci_build_qt6
        ;;
    cpu-noble-qt6)
        cpu-noble-qt6_export_env
        ci_build_qt6
        ;;
    cpu-focal-qt6-release)
        cpu-focal-qt6-release_export_env
        ci_build_qt6
        ;;
    cpu-jammy-qt6-release)
        cpu-jammy-qt6-release_export_env
        ci_build_qt6
        ;;
    cpu-noble-qt6-release)
        cpu-noble-qt6-release_export_env
        ci_build_qt6
        ;;

    # CUDA CI with Qt6
    cuda-focal-qt6)
        cuda-focal-qt6_export_env
        ci_build_qt6
        ;;
    cuda-jammy-qt6)
        cuda-jammy-qt6_export_env
        ci_build_qt6
        ;;
    cuda-noble-qt6)
        cuda-noble-qt6_export_env
        ci_build_qt6
        ;;
    cuda-focal-qt6-release)
        cuda-focal-qt6-release_export_env
        ci_build_qt6
        ;;
    cuda-jammy-qt6-release)
        cuda-jammy-qt6-release_export_env
        ci_build_qt6
        ;;
    cuda-noble-qt6-release)
        cuda-noble-qt6-release_export_env
        ci_build_qt6
        ;;

    *)
        echo "Error: invalid argument: ${1}." >&2
        print_usage_and_exit_docker_build_qt6
        ;;
    esac
}

# main() will be executed when ./docker_build_qt6.sh is called directly.
# main() will not be executed when ./docker_build_qt6.sh is sourced.
if [ "$0" = "$BASH_SOURCE" ]; then
    main "$@"
fi

