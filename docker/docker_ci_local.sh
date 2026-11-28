#!/usr/bin/env bash
#
# docker_ci_local.sh iterates on a CI configuration locally without pushing.
#
# `docker_build.sh` + `docker_test.sh <config>` reproduce a CI job exactly, but
# every round re-sends the build context and rebuilds the whole app inside
# `docker build` (hours cold). This script builds the CI image ONCE, keeps a
# long-lived container started from it, and then:
#   - syncs the host source tree over /root/ACloudViewer (the image's build
#     tree stays in place, so the next build is incremental),
#   - rebuilds only the requested targets,
#   - runs the same `run_cpp_unit_tests` entry docker_test.sh uses, or plain
#     ctest with a filter.
#
# The container keeps the toolchain, apt/Qt/VTK/PCL deps and ccache of the CI
# image, so what passes here is what the matching CI job compiles and runs.
# The cuda-<codename> image is the same Ubuntu release plus the CUDA toolkit:
# one image covers both the cpu- and the cuda- job of that release.
#
# Guidelines:
# - Same flat config names as docker_build.sh / docker_test.sh.

set -euo pipefail

print_usage_and_exit_docker_ci_local() {
    cat <<USAGE
USAGE:
    $(basename "$0") <config> <command> [args...]

CONFIG (same names as docker_build.sh / docker_test.sh):
    cpu-{focal,jammy,noble,resolute}[-release]      CPU CI build
    cuda-{focal,jammy,noble,resolute}[-release]     CUDA CI build (--gpus all when available)

COMMAND:
    up                    build the CI image when missing (BUILD_JOBS, default 16) and start the container
    sync                  copy the host source tree into the container (skipped: .git, build*, .ccache)
    build [targets...]    incremental cmake --build of the targets (all when none); full
                          build_gui_app.sh configure when the build tree does not exist yet
    test [ctest args...]  no args: the docker_test.sh profile (ci / ci,gpu);
                          args: ctest -C Release <args> inside build/, e.g.
                          test -R test_qanylabeling_chat --output-on-failure
    shell                 interactive bash in the container (/root/ACloudViewer)
    down                  remove the container (the image is kept)

build/test sync the source first. Sync overwrites files but never deletes: use
'down' + 'up' after removing or renaming sources.
USAGE
    exit 1
}

HOST_ACLOUDVIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

if [[ "$#" -lt 2 ]]; then
    print_usage_and_exit_docker_ci_local
fi

[[ "$1" =~ ^(cpu|cuda)-(focal|jammy|noble|resolute)(-release)?$ ]] || print_usage_and_exit_docker_ci_local
KIND="${BASH_REMATCH[1]}"
IMAGE="cloudviewer-ci:${KIND}-${BASH_REMATCH[2]}"
NAME="acv-ci-${KIND}-${BASH_REMATCH[2]}"
BUILD_CUDA_MODULE="OFF"
if [ "${KIND}" = "cuda" ]; then
    BUILD_CUDA_MODULE="ON"
fi
BUILD_JOBS="${BUILD_JOBS:-16}"
# Same in-image paths as Dockerfile.ci.
SRC=/root/ACloudViewer
BUILD=/root/ACloudViewer/build
cmd="$2"
shift 2

container_up() {
    if [ -z "$(docker images -q "${IMAGE}")" ]; then
        echo "[docker_ci_local] building ${IMAGE} (cold, same recipe as CI)"
        BUILD_JOBS="${BUILD_JOBS}" "${HOST_ACLOUDVIEWER_ROOT}/docker/docker_build.sh" "${KIND}-${BASH_REMATCH[2]}"
    fi
    if [ -z "$(docker ps -q -f "name=^${NAME}\$")" ]; then
        local gpu_args=()
        if [ "${BUILD_CUDA_MODULE}" = "ON" ] && docker info 2>/dev/null | grep -qi "runtimes.*nvidia"; then
            gpu_args=(--gpus all)
        fi
        docker run -d --name "${NAME}" "${gpu_args[@]}" "${IMAGE}" sleep infinity >/dev/null
        echo "[docker_ci_local] started ${NAME}"
    fi
}

source_sync() {
    # .git (3.9 GB) and the dependency archives are already in the image.
    tar -C "${HOST_ACLOUDVIEWER_ROOT}" \
        --exclude='./.git' --exclude='./build*' --exclude='./.ccache' \
        -cf - . | docker exec -i "${NAME}" tar -C "${SRC}" -xf -
}

in_container() {
    docker exec -i "${NAME}" bash -c "$1"
}

case "${cmd}" in
up)
    container_up
    ;;
sync)
    container_up
    source_sync
    ;;
build)
    container_up
    source_sync
    targets=()
    if [ "$#" -gt 0 ]; then
        targets=(--target "$@")
    fi
    if in_container "[ -f ${BUILD}/CMakeCache.txt ]"; then
        in_container "cmake --build ${BUILD} ${targets[*]+'${targets[*]}'} -j ${BUILD_JOBS}"
    else
        in_container "WITH_RDB=ON ./docker/build_gui_app.sh 3.12 ${BUILD_CUDA_MODULE}"
    fi
    ;;
test)
    container_up
    source_sync
    if [ "$#" -eq 0 ]; then
        profile=ci
        if [ -n "$(docker inspect "${NAME}" --format '{{.HostConfig.DeviceRequests}}' | tr -d '[]')" ]; then
            profile=ci,gpu
        fi
        in_container "source util/ci_utils.sh && cd build && \
            GTEST_FILTER='-*Reduce*Sum*' GTEST_SHUFFLE=1 run_cpp_unit_tests . ${profile}"
    else
        in_container "cd build && QT_QPA_PLATFORM=offscreen ctest -C Release $*"
    fi
    ;;
shell)
    container_up
    docker exec -it "${NAME}" bash
    ;;
down)
    if [ -z "$(docker ps -a -q -f "name=^${NAME}\$")" ]; then
        echo "[docker_ci_local] ${NAME} is not running"
    else
        docker rm -f "${NAME}" >/dev/null && echo "[docker_ci_local] removed ${NAME} (${IMAGE} kept)"
    fi
    ;;
*)
    print_usage_and_exit_docker_ci_local
    ;;
esac
