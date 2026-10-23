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
#   options. E.g., to support Ubuntu {20.04, 24.04} with Python {3.10, 3.13}, we
#   don't specify the OS and Python version separately, instead, we have a flat
#   list of combinations: [u2004_py310, u2004_py313, u2404_py310, u2404_py313].
# - No external environment variables.
#   This script should not make assumptions on external environment variables.
#   This make the Docker image reproducible across different machines.
set -euo pipefail

export BUILDKIT_PROGRESS=plain

__usage_docker_build="USAGE:
    $(basename $0) [OPTION]

OPTION:

    # Ubuntu CPU CI (Dockerfile.ci)
    cpu-focal                  : Ubuntu 20.04 CPU CI, developer mode
    cpu-jammy                  : Ubuntu 22.04 CPU CI, developer mode
    cpu-noble                  : Ubuntu 24.04 CPU CI, developer mode
    cpu-focal-release          : Ubuntu 20.04 CPU CI, release mode
    cpu-jammy-release          : Ubuntu 22.04 CPU CI, release mode
    cpu-noble-release          : Ubuntu 24.04 CPU CI, release mode

    # CUDA CI (Dockerfile.ci)
    cuda-focal                 : CUDA CI, Ubuntu 20.04, developer mode
    cuda-jammy                 : CUDA CI, Ubuntu 22.04, developer mode
    cuda-noble                 : CUDA CI, Ubuntu 24.04, developer mode
    cuda-focal-release         : CUDA CI, Ubuntu 20.04, release mode
    cuda-jammy-release         : CUDA CI, Ubuntu 22.04, release mode
    cuda-noble-release         : CUDA CI, Ubuntu 24.04, release mode

    # CUDA wheels (Dockerfile.ci)
    cuda_wheel_py310_dev       : CUDA Python 3.10 wheel, developer mode
    cuda_wheel_py311_dev       : CUDA Python 3.11 wheel, developer mode
    cuda_wheel_py312_dev       : CUDA Python 3.12 wheel, developer mode
    cuda_wheel_py313_dev       : CUDA Python 3.13 wheel, developer mode
    cuda_wheel_py310           : CUDA Python 3.10 wheel, release mode
    cuda_wheel_py311           : CUDA Python 3.11 wheel, release mode
    cuda_wheel_py312           : CUDA Python 3.12 wheel, release mode
    cuda_wheel_py313           : CUDA Python 3.13 wheel, release mode
"

HOST_CLOUDVIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

# Shared variables
CMAKE_VERSION=cmake-3.31.8-linux-x86_64
CMAKE_VERSION_AARCH64=cmake-3.31.8-linux-aarch64
CUDA_VERSION=12.6.3-cudnn
CUDA_VERSION_LATEST=12.6.3-cudnn
UBUNTU_FOCAL=20.04
UBUNTU_JAMMY=22.04
UBUNTU_NOBLE=24.04
UBUNTU_VERSION=$UBUNTU_JAMMY

print_usage_and_exit_docker_build() {
    echo "$__usage_docker_build"
    exit 1
}

cuda_wheel_build() {
    BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
    export BUILD_SHARED_LIBS=OFF

    QT_BASE_DIR="/usr/lib/x86_64-linux-gnu/qt5"

    options="$(echo "$@" | tr ' ' '|')"
    echo "[cuda_wheel_build()] options: ${options}"
    if [[ "py310" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.10
    elif [[ "py311" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.11
    elif [[ "py312" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.12
    elif [[ "py313" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.13
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
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
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

    # Copy wheel to host
    docker run -v "${PWD}:/opt/mount" --rm cloudviewer-ci:wheel \
        bash -c "cp /root/install/cloudviewer*.whl /opt/mount \
            && chown $(id -u):$(id -g) /opt/mount/cloudviewer*.whl"

    # Export the container's ccache directory for the workflow's
    # actions/cache post-step to save; same bind-mounted-file + ccache -p
    # resolution contract as ci_build (the NVIDIA ENTRYPOINT banner would
    # corrupt a stdout tar pipe).
    if docker run --rm -v "${PWD}:/opt/mount" cloudviewer-ci:wheel \
        bash -c "D=\$(ccache -p | awk '\$1 == \"cache_dir\" {print \$NF}') && ccache --show-stats > /opt/mount/.ccache-stats.txt || true && tar -C \"\$D\" -czf /opt/mount/.ccache-export.tgz ." \
       && mkdir -p "${PWD}/.ccache" && tar -C "${PWD}/.ccache" -xzf "${PWD}/.ccache-export.tgz"; then
        rm -f "${PWD}/.ccache-export.tgz"
        echo "[cuda_wheel_build()] ccache exported to ${PWD}/.ccache"
    else
        rm -f "${PWD}/.ccache-export.tgz"
        echo "[cuda_wheel_build()] WARNING: ccache export failed/skipped (next run builds cold)"
    fi
}

ci_build() {
    QT_BASE_DIR="/usr/lib/x86_64-linux-gnu/qt5"

    echo "[ci_build()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[ci_build()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[ci_build()] QT_BASE_DIR: ${QT_BASE_DIR}"
    echo "[ci_build()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[ci_build()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[ci_build()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[ci_build()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[ci_build()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[ci_build()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[ci_build()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"

    # Build parallelism: current hosted runners are 4 vCPU / 16 GB (runner
    # images 2026+), so CUDA builds default to 4 (nvcc peak ~2-4 GB per TU,
    # -j4 fits in 16 GB). The old 2 was tuned for the 2-core/7 GB era and
    # left half the CPUs idle. An explicit BUILD_JOBS override still wins.
    if [ "${BUILD_CUDA_MODULE:-OFF}" = "ON" ]; then
        BUILD_JOBS="${BUILD_JOBS:-4}"
    else
        BUILD_JOBS="${BUILD_JOBS:-4}"
    fi
    echo "[ci_build()] BUILD_JOBS=${BUILD_JOBS}"

    pushd "${HOST_CLOUDVIEWER_ROOT}"
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        --build-arg QT_BASE_DIR="${QT_BASE_DIR}" \
        --build-arg DEVELOPER_BUILD="${DEVELOPER_BUILD}" \
        --build-arg CMAKE_VERSION="${CMAKE_VERSION}" \
        --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
        --build-arg BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
        --build-arg BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}" \
        --build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
        --build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
        --build-arg BUILD_JOBS="${BUILD_JOBS}" \
        --build-arg PACKAGE="ON" \
        --build-arg CI="${CI:-}" \
        -t "${DOCKER_TAG}" \
        -f docker/Dockerfile.ci .
    popd

    docker run -v "${PWD}:/opt/mount" --rm "${DOCKER_TAG}" \
        bash -cx "cp /root/install/*run /opt/mount \
               && chown $(id -u):$(id -g) /opt/mount/*run"

    # Export the container's ccache directory (CCACHE_DIR=/root/ACloudViewer/.ccache,
    # see Dockerfile.ci) back into the workspace, on top of the copy the
    # actions/cache step restored before the build. The workflow's
    # actions/cache post-step then saves the updated tree for the next run.
    # Hosted runners boot a fresh VM every run, so the Docker layer cache is
    # always empty; the compiler cache in this directory is the only thing
    # that survives between runs. Best-effort: a missing/failed export must
    # not fail the build (the next run just recompiles cold).
    # The tarball goes through a bind-mounted FILE, never through the
    # container's stdout: the NVIDIA base image ENTRYPOINT prints a banner to
    # stdout on every docker run, which corrupts a streamed tar pipe (run
    # 39565766621 cuda-jammy: "tar: This does not look like a tar archive" ->
    # the cache was silently never saved and every run stayed cold).
    # ccache -p resolves the live cache_dir (Open3D's pattern) so the export
    # cannot drift from whatever CCACHE_DIR the image actually uses; the
    # archive stores the directory's CONTENTS so the on-disk name stays
    # .ccache/ regardless of that path. The \$ escapes keep ccache -p running
    # inside the container, not on the runner host. The awk matches the exact
    # cache_dir key only: focal's ccache 3.7.7 also prints
    # cache_dir_levels = 2, which the old /cache_dir/ substring match
    # absorbed into the path (tar -C '/root/.../.ccache\n2' -> export aborted,
    # measured in the 95622928 round).
    if docker run --rm -v "${PWD}:/opt/mount" "${DOCKER_TAG}" \
        bash -c "D=\$(ccache -p | awk '\$1 == \"cache_dir\" {print \$NF}') && ccache --show-stats > /opt/mount/.ccache-stats.txt || true && tar -C \"\$D\" -czf /opt/mount/.ccache-export.tgz ." \
       && mkdir -p "${PWD}/.ccache" && tar -C "${PWD}/.ccache" -xzf "${PWD}/.ccache-export.tgz"; then
        rm -f "${PWD}/.ccache-export.tgz"
        echo "[ci_build()] ccache exported to ${PWD}/.ccache (stats: .ccache-stats.txt)"
    else
        rm -f "${PWD}/.ccache-export.tgz"
        echo "[ci_build()] WARNING: ccache export failed/skipped (next run builds cold)"
    fi
}

cuda-focal_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-focal

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-jammy_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-jammy

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-noble_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-noble

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-focal-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-focal

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-jammy-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-jammy

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cuda-noble-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cuda-noble

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=nvidia/cuda:${CUDA_VERSION_LATEST}-devel-ubuntu${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-focal_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-focal

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-jammy_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-jammy

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-noble_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-noble

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=ON
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-focal-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-focal

    export UBUNTU_VERSION=${UBUNTU_FOCAL}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-jammy-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-jammy

    export UBUNTU_VERSION=${UBUNTU_JAMMY}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=OFF
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
}

cpu-noble-release_export_env() {
    export DOCKER_TAG=cloudviewer-ci:cpu-noble

    export UBUNTU_VERSION=${UBUNTU_NOBLE}
    export BASE_IMAGE=ubuntu:${UBUNTU_VERSION}
    export DEVELOPER_BUILD=OFF
    export PYTHON_VERSION=3.12
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
    cpu-noble)
        cpu-noble_export_env
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
    cpu-noble-release)
        cpu-noble-release_export_env
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
    cuda-noble)
        cuda-noble_export_env
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
    cuda-noble-release)
        cuda-noble-release_export_env
        ci_build
        ;;

    # CUDA wheels
    cuda_wheel_py310_dev)
        cuda_wheel_build py310 dev
        ;;
    cuda_wheel_py311_dev)
        cuda_wheel_build py311 dev
        ;;
    cuda_wheel_py312_dev)
        cuda_wheel_build py312 dev
        ;;
    cuda_wheel_py313_dev)
        cuda_wheel_build py313 dev
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
    cuda_wheel_py313)
        cuda_wheel_build py313
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
