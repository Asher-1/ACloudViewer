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
    # CUDA wheels (Dockerfile.ci)
    cuda_wheel_py310_dev        : CUDA Python 3.10 wheel, developer mode
    cuda_wheel_py311_dev        : CUDA Python 3.11 wheel, developer mode
    cuda_wheel_py312_dev        : CUDA Python 3.12 wheel, developer mode
    cuda_wheel_py313_dev        : CUDA Python 3.13 wheel, developer mode
    cuda_wheel_py310            : CUDA Python 3.10 wheel, release mode
    cuda_wheel_py311            : CUDA Python 3.11 wheel, release mode
    cuda_wheel_py312            : CUDA Python 3.12 wheel, release mode
    cuda_wheel_py313            : CUDA Python 3.13 wheel, release mode

    # Qt6 builds (Dockerfile.ci.qt6)
    qt6-cpu                     : Qt6 CPU build
    qt6-cuda                    : Qt6 CUDA build

    # CI builds (Dockerfile.ci) - matches CI_CONFIG from GitHub Actions
    cpu-focal                   : CPU Ubuntu 20.04 CI build
    cpu-jammy                   : CPU Ubuntu 22.04 CI build
    cpu-noble                   : CPU Ubuntu 24.04 CI build
    cpu-focal-release           : CPU Ubuntu 20.04 CI build, release mode
    cpu-jammy-release           : CPU Ubuntu 22.04 CI build, release mode
    cpu-noble-release           : CPU Ubuntu 24.04 CI build, release mode
    cuda-focal                  : CUDA Ubuntu 20.04 CI build
    cuda-jammy                  : CUDA Ubuntu 22.04 CI build
    cuda-noble                  : CUDA Ubuntu 24.04 CI build
    cuda-focal-release          : CUDA Ubuntu 20.04 CI build, release mode
    cuda-jammy-release          : CUDA Ubuntu 22.04 CI build, release mode
    cuda-noble-release          : CUDA Ubuntu 24.04 CI build, release mode
"

HOST_ACLOUDVIEWER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd)"

print_usage_and_exit_docker_test() {
    echo "$__usage_docker_test"
    exit 1
}

ci_print_env() {
    echo "[ci_print_env()] DOCKER_TAG=${DOCKER_TAG:-not set}"
    echo "[ci_print_env()] DEVELOPER_BUILD=${DEVELOPER_BUILD:-not set}"
    echo "[ci_print_env()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS:-not set}"
    echo "[ci_print_env()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE:-not set}"
    echo "[ci_print_env()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:-not set}"
    echo "[ci_print_env()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:-not set}"
}

# Whole-tree python unit tests, run inside the container under test. The
# container image installs the python package (with the reconstruction
# bindings), so pytest python/test collects the reconstruction
# pycolmap-parity suite together with everything else - no special-casing.
python_unit_tests() {
    echo "======================================================================"
    echo "Running Python Unit Tests"
    echo "======================================================================"
    echo "pytest is randomized, add --randomly-seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c '
        # The wheel images ship pytest; the CI images do not.
        python -m pip install --quiet pytest >/dev/null 2>&1 || true
        # The CI images never pip-install the cloudViewer package: the GUI-app
        # build assembles it (pybind target) under
        # $ACloudViewer_BUILD/lib/python_package (exact case as defined by
        # Dockerfile.ci), which nothing puts on sys.path. Export it when the
        # import would otherwise fail. The wheel images pip-install the
        # package and wipe their build tree, so they take the first branch
        # and stay untouched.
        if ! python -c "import cloudViewer" 2>/dev/null; then
            if [ -n "${ACloudViewer_BUILD:-}" ] && [ -f "${ACloudViewer_BUILD}/CMakeCache.txt" ]; then
                if [ ! -d "${ACloudViewer_BUILD}/lib/python_package/cloudViewer" ]; then
                    # The GUI-app image build stops at `make install`: the
                    # python-package assembly target (`make python-package`)
                    # is separate and never runs there. Build it on demand so
                    # the self-loading package exists for pytest.
                    if ! cmake --build "${ACloudViewer_BUILD}" --target python-package; then
                        if [ "${BUILD_CUDA_MODULE:-OFF}" = "ON" ]; then
                            # CUDA-only GUI builds stage a single (CUDA)
                            # pybind variant; the cpu fallback variant that
                            # the python-package guard requires is only
                            # produced by the two-pass wheel build, so the
                            # target cannot complete here. Python coverage
                            # stays on the cpu images and the wheel path.
                            echo "INFO: CUDA-only GUI build lacks the CPU pybind fallback; skipping python tests."
                            exit 0
                        fi
                        echo "ERROR: python-package assembly failed in ${ACloudViewer_BUILD}" >&2
                        exit 1
                    fi
                fi
                export PYTHONPATH="${ACloudViewer_BUILD}/lib/python_package${PYTHONPATH:+:$PYTHONPATH}"
            else
                echo "ERROR: cloudViewer not importable and no build tree at ${ACloudViewer_BUILD:-<unset>} to assemble it from" >&2
                exit 1
            fi
            # Whole-tree tests also need the runtime deps (numpy, plyfile,
            # ...) and test-only deps (scipy, tensorboard, pillow, ...) that
            # the wheel environments provide but the CI images do not.
            python -m pip install --quiet -r python/requirements.txt \
                -r python/requirements_test.txt || exit 1
        fi
        # Check if ML ops should be tested by checking build configuration.
        pytest_args=""
        if ! python -c "import sys, cloudViewer; sys.exit(not (cloudViewer._build_config[\"BUILD_PYTORCH_OPS\"] or cloudViewer._build_config[\"BUILD_TENSORFLOW_OPS\"]))" 2>/dev/null; then
            pytest_args="--ignore python/test/ml_ops/"
        fi
        python -W default -m pytest python/test ${pytest_args} -v
    '
}

cpp_test_only() {
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_CUDA_MODULE
    # - NPROC (optional)
    echo "[cpp_test_only()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[cpp_test_only()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[cpp_test_only()] NPROC=${NPROC:=$(nproc)}"

    # Config-dependent argument: gpu_run_args
    docker_run="docker run --cpus ${NPROC}"
    if [ "${BUILD_CUDA_MODULE}" == "ON" ]; then
        # Check if GPU is available before adding --gpus flag
        # This allows CUDA builds to run on systems without GPU (tests will skip GPU-specific tests)
        # Check if nvidia-container-toolkit is available by checking docker info or nvidia-smi
        if (docker info 2>/dev/null | grep -qi "runtimes.*nvidia") || \
           (command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1); then
            docker_run="${docker_run} --gpus all"
            echo "[cpp_test_only()] GPU support detected, using --gpus all"
        else
            echo "[cpp_test_only()] Warning: GPU not available, running without --gpus flag"
            echo "[cpp_test_only()] CUDA tests that require GPU will be skipped"
        fi
    fi

    # C++ test only
    echo "======================================================================"
    echo "Running C++ Unit Tests"
    echo "======================================================================"
    echo "gtest is randomized, add --gtest_random_seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm ${DOCKER_TAG} /bin/bash -c " \
        export BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE} \
     && cd build \
     && ./bin/cloudviewer_test/tests --gtest_shuffle --gtest_filter=-*Reduce*Sum* \
     && source ../util/ci_utils.sh \
     && run_qpythonruntime_tests . native \
     && cmake --build . --target reconstruction_oiio_parity_gate -j2 \
     && cmake --build . --target reconstruction_camera_rig_parity_gate -j2 \
     && if cmake -L -N . | grep -q '^RECONSTRUCTION_CASPAR_ENABLED:BOOL=ON$'; then \
            cmake --build . --target reconstruction_caspar_parity_gate -j2; \
        fi \
     && cmake --build . --target reconstruction_cpp_tests_gate -j2 \
     && if test -f bin/libAICore.so; then \
            python ../util/check_aicore_runtime.py bin/libAICore.so; \
        fi \
    "

    # Python unit tests (whole python/test tree). The CI image installs the
    # python package (build_gui_app: BUILD_PYTHON_MODULE=ON + BUILD_
    # RECONSTRUCTION=ON + make install), so this covers the reconstruction
    # pycolmap-parity suite as well - no separate compile anywhere.
    python_unit_tests
}

cpp_python_command_tools_test() {
    # Expects the following environment variables to be set:
    # - DOCKER_TAG
    # - BUILD_SHARED_LIBS
    # - BUILD_CUDA_MODULE
    # - BUILD_PYTORCH_OPS (for logging only, actual check is done dynamically)
    # - BUILD_TENSORFLOW_OPS (for logging only, actual check is done dynamically)
    # - NPROC (optional)
    echo "[cpp_python_command_tools_test()] DOCKER_TAG=${DOCKER_TAG}"
    echo "[cpp_python_command_tools_test()] BUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
    echo "[cpp_python_command_tools_test()] BUILD_CUDA_MODULE=${BUILD_CUDA_MODULE}"
    echo "[cpp_python_command_tools_test()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS}"
    echo "[cpp_python_command_tools_test()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS}"
    echo "[cpp_python_command_tools_test()] NPROC=${NPROC:=$(nproc)}"
    echo "[cpp_python_command_tools_test()] Note: ML ops testing is determined dynamically by checking cloudViewer._build_config"

    # Config-dependent argument: gpu_run_args
    docker_run="docker run --cpus ${NPROC}"
    if [ "${BUILD_CUDA_MODULE}" == "ON" ]; then
        # Check if GPU is available before adding --gpus flag
        # This allows CUDA builds to run on systems without GPU (tests will skip GPU-specific tests)
        # Check if nvidia-container-toolkit is available by checking docker info or nvidia-smi
        if (docker info 2>/dev/null | grep -qi "runtimes.*nvidia") || \
           (command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1); then
            docker_run="${docker_run} --gpus all"
            echo "[cpp_python_command_tools_test()] GPU support detected, using --gpus all"
        else
            echo "[cpp_python_command_tools_test()] Warning: GPU not available, running without --gpus flag"
            echo "[cpp_python_command_tools_test()] CUDA tests that require GPU will be skipped"
        fi
    fi

    # C++ test
    echo "======================================================================"
    echo "Running C++ Unit Tests"
    echo "======================================================================"
    echo "gtest is randomized, add --gtest_random_seed=SEED to repeat the test sequence."
    ${docker_run} -i --rm ${DOCKER_TAG} /bin/bash -c " \
        cd build \
     && ./bin/cloudviewer_test/tests --gtest_shuffle --gtest_filter=-*Reduce*Sum* \
     && source ../util/ci_utils.sh \
     && run_qpythonruntime_tests . native \
     && cmake --build . --target reconstruction_oiio_parity_gate -j2 \
     && cmake --build . --target reconstruction_camera_rig_parity_gate -j2 \
     && if cmake -L -N . | grep -q '^RECONSTRUCTION_CASPAR_ENABLED:BOOL=ON$'; then \
            cmake --build . --target reconstruction_caspar_parity_gate -j2; \
        fi \
     && if test -f bin/libAICore.so; then \
            python ../util/check_aicore_runtime.py bin/libAICore.so; \
        fi \
    "

    # Python test (shared whole-tree runner; the wheel images install the
    # python package, so this includes the reconstruction parity suite)
    python_unit_tests

    # Command-line tools test
    echo "======================================================================"
    echo "Testing ACloudViewer command-line tools"
    echo "======================================================================"
    ${docker_run} -i --rm "${DOCKER_TAG}" /bin/bash -c "\
        cloudviewer \
     && cloudviewer -h \
     && cloudviewer --help \
     && cloudviewer -V \
     && cloudviewer --version \
     && cloudviewer example -h \
     && cloudviewer example --help \
     && cloudviewer example -l \
     && cloudviewer example --list \
     && cloudviewer example -l io \
     && cloudviewer example --list io \
     && cloudviewer example -s io/image_io \
     && cloudviewer example --show io/image_io \
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

# CUDA wheels
cuda_wheel_py310_dev)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
cuda_wheel_py311_dev)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
cuda_wheel_py312_dev)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
cuda_wheel_py313_dev)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
cuda_wheel_py310)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
cuda_wheel_py311)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
cuda_wheel_py312)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
cuda_wheel_py313)
    export DOCKER_TAG="cloudviewer-ci:wheel"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="ON"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;

# Qt6 builds
qt6-cpu)
    export DOCKER_TAG="acloudviewer-qt6-cpu:latest"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;
qt6-cuda)
    export DOCKER_TAG="acloudviewer-qt6-cuda:latest"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_python_command_tools_test
    ;;

# CI builds (matches CI_CONFIG from GitHub Actions)
cpu-focal)
    export DOCKER_TAG="cloudviewer-ci:cpu-focal"
    export DEVELOPER_BUILD="ON"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cpu-jammy)
    export DOCKER_TAG="cloudviewer-ci:cpu-jammy"
    export DEVELOPER_BUILD="ON"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cpu-noble)
    export DOCKER_TAG="cloudviewer-ci:cpu-noble"
    export DEVELOPER_BUILD="ON"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cpu-focal-release)
    export DOCKER_TAG="cloudviewer-ci:cpu-focal"
    export DEVELOPER_BUILD="OFF"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cpu-jammy-release)
    export DOCKER_TAG="cloudviewer-ci:cpu-jammy"
    export DEVELOPER_BUILD="OFF"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cpu-noble-release)
    export DOCKER_TAG="cloudviewer-ci:cpu-noble"
    export DEVELOPER_BUILD="OFF"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="OFF"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cuda-focal)
    export DOCKER_TAG="cloudviewer-ci:cuda-focal"
    export DEVELOPER_BUILD="ON"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cuda-jammy)
    export DOCKER_TAG="cloudviewer-ci:cuda-jammy"
    export DEVELOPER_BUILD="ON"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cuda-noble)
    export DOCKER_TAG="cloudviewer-ci:cuda-noble"
    export DEVELOPER_BUILD="ON"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cuda-focal-release)
    export DOCKER_TAG="cloudviewer-ci:cuda-focal"
    export DEVELOPER_BUILD="OFF"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cuda-jammy-release)
    export DOCKER_TAG="cloudviewer-ci:cuda-jammy"
    export DEVELOPER_BUILD="OFF"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
cuda-noble-release)
    export DOCKER_TAG="cloudviewer-ci:cuda-noble"
    export DEVELOPER_BUILD="OFF"
    export BUILD_SHARED_LIBS="OFF"
    export BUILD_CUDA_MODULE="ON"
    export BUILD_PYTORCH_OPS="OFF"
    export BUILD_TENSORFLOW_OPS="OFF"
    ci_print_env
    cpp_test_only
    ;;
*)
    echo "Error: invalid argument: ${1}." >&2
    print_usage_and_exit_docker_test
    ;;
esac
