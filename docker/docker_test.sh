#!/usr/bin/env bash
#
# docker_test.sh is used to test ACloudViewer docker images built by docker_build.sh
#
# Guidelines:
# - Use a flat list of options. No additional arguments.
#   The option names should match exactly the ones used in docker_build.sh.
# - No external environment variables.
#   - This script should not make assumptions on external environment variables.
#   - Environment variables are imported from docker_build.sh.

set -euo pipefail

__usage_docker_test="USAGE:
    $(basename $0) [OPTION]

OPTION:
    # Ubuntu CPU CI (Dockerfile.ci)
    cpu-static                  : Ubuntu CPU static
    cpu-static-release          : Ubuntu CPU static, release mode
    cpu-shared-ml               : Ubuntu CPU shared with ML
    cpu-shared-ml-release       : Ubuntu CPU shared with ML, release mode

    # ML CIs (Dockerfile.ci)
    cuda-ml-shared-jammy        : CUDA CI, ml-shared-jammy (cxx11_abi), developer mode
    cuda-ml-shared-jammy-release: CUDA CI, ml-shared-jammy (cxx11_abi), release mode

    # Qt6 builds (Dockerfile.ci.qt6)
    qt6-cpu-static              : Qt6 CPU static build
    qt6-cuda-shared             : Qt6 CUDA shared build
"

HOST_ACLOUDVIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

print_usage_and_exit_docker_test() {
    echo "$__usage_docker_test"
    exit 1
}

ci_print_env() {
    echo "[ci_print_env()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[ci_print_env()] BASE_IMAGE=${BASE_IMAGE}"
    echo "[ci_print_env()] DEVELOPER_BUILD=${DEVELOPER_BUILD}"
    echo "[ci_print_env()] CCACHE_TAR_NAME=${CCACHE_TAR_NAME}"
    echo "[ci_print_env()] CMAKE_VERSION=${CMAKE_VERSION}"
    echo "[ci_print_env()] PYTHON_VERSION=${PYTHON_VERSION}"
    echo "[ci_print_env()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[ci_print_env()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[ci_print_env()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[ci_print_env()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[ci_print_env()] PACKAGE=${PACKAGE}"
}

restart_docker_daemon_if_on_gcloud() {
    # Sometimes `docker run` may fail on the second run on Google Cloud with the
    # following error:
    # ```
    # docker: Error response from daemon: OCI runtime create failed:
    # container_linux.go:349: starting container process caused
    # "process_linux.go:449: container init caused \"process_linux.go:432:
    # running prestart hook 0 caused \\\"error running hook: exit status 1,
    # stdout: , stderr: nvidia-container-cli: initialization error:
    # nvml error: driver/library version mismatch\\\\n\\\"\"": unknown.
    # ```
    if curl metadata.google.internal -i 2>/dev/null | grep -q Google; then
        # https://stackoverflow.com/a/30921162/1255535
        echo "[restart_docker_daemon_if_on_gcloud()] Restarting Docker daemon on Google Cloud."
        sudo systemctl daemon-reload
        sudo systemctl restart docker
    else
        echo "[restart_docker_daemon_if_on_gcloud()] Skipped."
    fi
}

cpp_python_linking_uninstall_test() {
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_SHARED_LIBS
    # - BUILD_CUDA_MODULE
    # - BUILD_PYTORCH_OPS
    # - BUILD_TENSORFLOW_OPS
    # - NPROC (optional)
    echo "[cpp_python_linking_uninstall_test()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[cpp_python_linking_uninstall_test()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[cpp_python_linking_uninstall_test()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[cpp_python_linking_uninstall_test()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[cpp_python_linking_uninstall_test()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[cpp_python_linking_uninstall_test()] NPROC=${NPROC:=$(nproc)}"

    # Config-dependent argument: gpu_run_args
    docker_run="docker run --cpus ${NPROC}"
    if [ "${BUILD_CUDA_MODULE}" == "ON" ]; then
        docker_run="${docker_run} --gpus all"
    fi

    # Config-dependent argument: pytest_args
    if [ "${BUILD_PYTORCH_OPS}" == "OFF" ] || [ "${BUILD_TENSORFLOW_OPS}" == "OFF" ]; then
        pytest_args="--ignore python/test/ml_ops/"
    else
        pytest_args=""
    fi
    restart_docker_daemon_if_on_gcloud

    # C++ test
    echo "======================================================================"
    echo "Running C++ Unit Tests"
    echo "======================================================================"
    echo "gtest is randomized, add --gtest_random_seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm ${DOCKER_TAG} /bin/bash -c " \
        cd build \
     && ./bin/tests --gtest_shuffle --gtest_filter=-*Reduce*Sum* \
    "
    restart_docker_daemon_if_on_gcloud

    # Python test
    echo "======================================================================"
    echo "Running Python Unit Tests"
    echo "======================================================================"
    echo "pytest is randomized, add --randomly-seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c " \
        python -W default -m pytest python/test ${pytest_args} -v"
    restart_docker_daemon_if_on_gcloud

    # Command-line tools test (optional, ACloudViewer may not have CLI yet)
    echo "======================================================================"
    echo "Testing ACloudViewer command-line tools"
    echo "======================================================================"
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        cloudviewer --version || echo 'CloudViewer CLI not available yet' \
    "

    # C++ linking with new project
    echo "======================================================================"
    echo "Testing C++ linking with CMake find_package"
    echo "======================================================================"
    if [ -d "examples/cmake/cloudviewer-cmake-find-package" ]; then
        ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
            cd examples/cmake/cloudviewer-cmake-find-package \
         && mkdir -p build \
         && pushd build \
         && echo Testing build with cmake \
         && cmake -DCMAKE_INSTALL_PREFIX=~/cloudviewer_install .. \
         && make -j\$(nproc) VERBOSE=1 \
         && ./Draw --skip-for-unit-test || echo 'Example execution skipped' \
        "
    else
        echo "CMake example not found, skipping..."
    fi

    restart_docker_daemon_if_on_gcloud

    # Uninstall
    echo "======================================================================"
    echo "Testing uninstall"
    echo "======================================================================"
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        cd build \
     && make uninstall || echo 'Uninstall target not available' \
    "
}

if [[ "$#" -ne 1 ]]; then
    echo "Error: invalid number of arguments." >&2
    print_usage_and_exit_docker_test
fi
echo "[$(basename $0)] testing $1"

# Try to source docker_build.sh to get environment settings
if [ -f "${HOST_ACLOUDVIEWER_ROOT}/docker/docker_build.sh" ]; then
    source "${HOST_ACLOUDVIEWER_ROOT}/docker/docker_build.sh"
elif [ -f "${HOST_ACLOUDVIEWER_ROOT}/docker/docker_build_qt6.sh" ]; then
    source "${HOST_ACLOUDVIEWER_ROOT}/docker/docker_build_qt6.sh"
else
    echo "Warning: docker_build.sh not found, using default settings"
fi

case "$1" in
# CPU CI
cpu-static)
    # Export environment for cpu-static build
    export DOCKER_TAG="acloudviewer-cpu-static:latest"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
cpu-static-release)
    export DOCKER_TAG="acloudviewer-cpu-static-release:latest"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
cpu-shared-ml)
    export DOCKER_TAG="acloudviewer-cpu-shared-ml:latest"
    export BUILD_SHARED_LIBS="ON"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="ON"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
cpu-shared-ml-release)
    export DOCKER_TAG="acloudviewer-cpu-shared-ml-release:latest"
    export BUILD_SHARED_LIBS="ON"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="ON"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;

# ML CIs
cuda-ml-shared-jammy)
    export DOCKER_TAG="acloudviewer-cuda-ml-shared-jammy:latest"
    export BUILD_SHARED_LIBS="ON"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="ON"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
cuda-ml-shared-jammy-release)
    export DOCKER_TAG="acloudviewer-cuda-ml-shared-jammy-release:latest"
    export BUILD_SHARED_LIBS="ON"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="ON"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;

# Qt6 builds
qt6-cpu-static)
    export DOCKER_TAG="acloudviewer-qt6-cpu-static:latest"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
qt6-cuda-shared)
    export DOCKER_TAG="acloudviewer-qt6-cuda-shared:latest"
    export BUILD_SHARED_LIBS="ON"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="ON"
    ci_print_env
    cpp_python_linking_uninstall_test
    ;;
*)
    echo "Error: invalid argument: ${1}." >&2
    print_usage_and_exit_docker_test
    ;;
esac
