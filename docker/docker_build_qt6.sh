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
    cpu-jammy-qt6             : Ubuntu 22.04 CPU with Qt6, developer mode
    cpu-noble-qt6             : Ubuntu 24.04 CPU with Qt6, developer mode
    cpu-oracular-qt6          : Ubuntu 26.04 CPU with Qt6, developer mode
    cpu-jammy-qt6-release     : Ubuntu 22.04 CPU with Qt6, release mode
    cpu-noble-qt6-release     : Ubuntu 24.04 CPU with Qt6, release mode
    cpu-oracular-qt6-release  : Ubuntu 26.04 CPU with Qt6, release mode

    # CUDA CI with Qt6 (Dockerfile.ci.qt6)
    cuda-jammy-qt6            : CUDA Ubuntu 22.04 with Qt6, developer mode
    cuda-noble-qt6            : CUDA Ubuntu 24.04 with Qt6, developer mode
    cuda-oracular-qt6         : CUDA Ubuntu 26.04 with Qt6, developer mode
    cuda-jammy-qt6-release    : CUDA Ubuntu 22.04 with Qt6, release mode
    cuda-noble-qt6-release    : CUDA Ubuntu 24.04 with Qt6, release mode
    cuda-oracular-qt6-release : CUDA Ubuntu 26.04 with Qt6, release mode

    # CUDA wheels with Qt6 (Dockerfile.wheel.qt6)
    cuda_wheel_qt6_py310_dev  : CUDA Python 3.10 wheel with Qt6, developer mode
    cuda_wheel_qt6_py311_dev  : CUDA Python 3.11 wheel with Qt6, developer mode
    cuda_wheel_qt6_py312_dev  : CUDA Python 3.12 wheel with Qt6, developer mode
    cuda_wheel_qt6_py313_dev  : CUDA Python 3.13 wheel with Qt6, developer mode
    cuda_wheel_qt6_py310      : CUDA Python 3.10 wheel with Qt6, release mode
    cuda_wheel_qt6_py311      : CUDA Python 3.11 wheel with Qt6, release mode
    cuda_wheel_qt6_py312      : CUDA Python 3.12 wheel with Qt6, release mode
    cuda_wheel_qt6_py313      : CUDA Python 3.13 wheel with Qt6, release mode

Build dependencies:
    - VTK 9.4.2
    - PCL 1.15.1
    - Qt 6.x (system packages from Ubuntu repository)
"

HOST_CLOUDVIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# Shared variables
CMAKE_VERSION=cmake-3.29.2-linux-x86_64
CUDA_VERSION=12.6.3-cudnn
CUDA_VERSION_LATEST=12.6.3-cudnn
UBUNTU_JAMMY=22.04
UBUNTU_NOBLE=24.04
UBUNTU_ORACULAR=26.04

print_usage_and_exit_docker_build_qt6() {
    echo "$__usage_docker_build_qt6"
    exit 1
}

# Check if Docker image exists locally
docker_image_exists() {
    local image="$1"
    docker image inspect "$image" >/dev/null 2>&1
}

# Pull Docker image with retry logic
docker_pull_with_retry() {
    local image="$1"
    local max_attempts=3
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        echo "[docker_pull_with_retry] Attempt $attempt/$max_attempts: Pulling ${image}..."
        if docker pull "$image"; then
            echo "[docker_pull_with_retry] Successfully pulled ${image}"
            return 0
        else
            if [ $attempt -lt $max_attempts ]; then
                echo "[docker_pull_with_retry] Failed to pull ${image}, retrying in 5 seconds..."
                sleep 5
            else
                echo "[docker_pull_with_retry] Failed to pull ${image} after $max_attempts attempts"
                return 1
            fi
        fi
        attempt=$((attempt + 1))
    done
}

# Ensure base image is available (local or pulled)
ensure_base_image() {
    local base_image="$1"
    
    if docker_image_exists "$base_image"; then
        echo "[ensure_base_image] Base image ${base_image} exists locally"
        return 0
    else
        echo "[ensure_base_image] Base image ${base_image} not found locally, attempting to pull..."
        if docker_pull_with_retry "$base_image"; then
            return 0
        else
            echo "[ensure_base_image] ERROR: Could not pull ${base_image}. Please check your network connection or Docker configuration."
            echo "[ensure_base_image] You can try manually: docker pull ${base_image}"
            return 1
        fi
    fi
}

# CUDA wheel build with Qt6
cuda_wheel_build_qt6() {
    UBUNTU_VERSION=${UBUNTU_JAMMY}
    BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF

    # Qt6 base directory for apt-installed Qt6
    QT_BASE_DIR="/usr/lib/x86_64-linux-gnu/qt6"

    options="$(echo "$@" | tr ' ' '|')"
    echo "[cuda_wheel_build_qt6()] options: ${options}"
    if [[ "py310" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.10
    elif [[ "py311" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.11
        if [ "${BUILD_CUDA_MODULE}" = "ON" ]; then
            # Disable PyTorch ops for Python 3.11 with CUDA due to pytorch issue
            export BUILD_PYTORCH_OPS=OFF
        fi
    elif [[ "py312" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.12
    elif [[ "py313" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.13
    else
        echo "Invalid python version."
        print_usage_and_exit_docker_build_qt6
    fi
    if [[ "dev" =~ ^($options)$ ]]; then
        DEVELOPER_BUILD=ON
    else
        DEVELOPER_BUILD=OFF
    fi
    echo "[cuda_wheel_build_qt6()] PYTHON_VERSION: ${PYTHON_VERSION}"
    echo "[cuda_wheel_build_qt6()] QT_BASE_DIR: ${QT_BASE_DIR}"
    echo "[cuda_wheel_build_qt6()] UBUNTU_VERSION: ${UBUNTU_VERSION}"
    echo "[cuda_wheel_build_qt6()] DEVELOPER_BUILD: ${DEVELOPER_BUILD}"
    echo "[cuda_wheel_build_qt6()] BUILD_SHARED_LIBS: ${BUILD_SHARED_LIBS}"
    echo "[cuda_wheel_build_qt6()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[cuda_wheel_build_qt6()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[cuda_wheel_build_qt6()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[cuda_wheel_build_qt6()] Using Qt6"

    # Ensure base image is available before building
    if ! ensure_base_image "${BASE_IMAGE}"; then
        echo "[cuda_wheel_build_qt6()] ERROR: Failed to ensure base image ${BASE_IMAGE}"
        return 1
    fi

    pushd "${HOST_CLOUDVIEWER_ROOT}"
    docker build \
        --network host \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg QT_BASE_DIR="${QT_BASE_DIR}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}" \
        --build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        --build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        --build-arg BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        --build-arg PACKAGE="OFF" \
        --build-arg CI="${CI:-}" \
        -t cloudviewer-ci:wheel-qt6 \
        -f docker/Dockerfile.wheel.qt6 .
    popd

    # Copy wheel to host
    docker run -v "${PWD}:/opt/mount" --rm cloudviewer-ci:wheel-qt6 \
        bash -c "cp /root/install/cloudviewer*.whl /opt/mount \
            && chown $(id -u):$(id -g) /opt/mount/cloudviewer*.whl"
}

ci_build_qt6() {
    # Set QT_BASE_DIR for apt-installed Qt6 (Ubuntu 22.04+):
    #   - QT_BASE_DIR=/usr/lib/x86_64-linux-gnu/qt6
    #   - lib: /usr/lib/x86_64-linux-gnu (NOT ${QT_BASE_DIR}/lib)
    #   - plugins: ${QT_BASE_DIR}/plugins
    #   - bin: /usr/lib/qt6/bin
    #
    # Note: For apt-installed Qt6, the directory structure is non-standard,
    # so we set QT_BASE_DIR to the qt6 subdirectory where plugins are located.
    # The actual lib path is handled separately in Dockerfile.ci.qt6
    QT_BASE_DIR="/usr/lib/x86_64-linux-gnu/qt6"

    echo "[ci_build_qt6()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[ci_build_qt6()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[ci_build_qt6()] UBUNTU_VERSION=${UBUNTU_VERSION}"
    echo "[ci_build_qt6()] QT_BASE_DIR=${QT_BASE_DIR}"
    echo "[ci_build_qt6()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[ci_build_qt6()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[ci_build_qt6()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[ci_build_qt6()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[ci_build_qt6()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[ci_build_qt6()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[ci_build_qt6()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[ci_build_qt6()] Using VTK 9.4.2, PCL 1.15.1, Qt6"

    # Ensure base image is available before building
    if ! ensure_base_image "${BASE_IMAGE}"; then
        echo "[ci_build_qt6()] ERROR: Failed to ensure base image ${BASE_IMAGE}"
        return 1
    fi

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

cpu-oracular-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-oracular-qt6

    export UBUNTU_VERSION=${UBUNTU_ORACULAR}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-oracular-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-oracular-qt6

    export UBUNTU_VERSION=${UBUNTU_ORACULAR}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

# CUDA builds with Qt6
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

cuda-oracular-qt6_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-oracular-qt6

    export UBUNTU_VERSION=${UBUNTU_ORACULAR}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-oracular-qt6-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-oracular-qt6

    export UBUNTU_VERSION=${UBUNTU_ORACULAR}
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
    cpu-jammy-qt6)
        cpu-jammy-qt6_export_env
        ci_build_qt6
        ;;
    cpu-noble-qt6)
        cpu-noble-qt6_export_env
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
    cpu-oracular-qt6)
        cpu-oracular-qt6_export_env
        ci_build_qt6
        ;;
    cpu-oracular-qt6-release)
        cpu-oracular-qt6-release_export_env
        ci_build_qt6
        ;;

    # CUDA CI with Qt6
    cuda-jammy-qt6)
        cuda-jammy-qt6_export_env
        ci_build_qt6
        ;;
    cuda-noble-qt6)
        cuda-noble-qt6_export_env
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
    cuda-oracular-qt6)
        cuda-oracular-qt6_export_env
        ci_build_qt6
        ;;
    cuda-oracular-qt6-release)
        cuda-oracular-qt6-release_export_env
        ci_build_qt6
        ;;

    # CUDA wheels with Qt6
    cuda_wheel_qt6_py310_dev)
        cuda_wheel_build_qt6 py310 dev
        ;;
    cuda_wheel_qt6_py311_dev)
        cuda_wheel_build_qt6 py311 dev
        ;;
    cuda_wheel_qt6_py312_dev)
        cuda_wheel_build_qt6 py312 dev
        ;;
    cuda_wheel_qt6_py313_dev)
        cuda_wheel_build_qt6 py313 dev
        ;;
    cuda_wheel_qt6_py310)
        cuda_wheel_build_qt6 py310
        ;;
    cuda_wheel_qt6_py311)
        cuda_wheel_build_qt6 py311
        ;;
    cuda_wheel_qt6_py312)
        cuda_wheel_build_qt6 py312
        ;;
    cuda_wheel_qt6_py313)
        cuda_wheel_build_qt6 py313
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

