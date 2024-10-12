#!/usr/bin/env bash
set -euo pipefail

# this scripts only used on macos
if [ -z "$CONDA_PREFIX" ] ; then
	echo "Conda env is not activated"
	exit -1
fi

# Get build scripts and control environment variables
# shellcheck source=ci_utils.sh
source "$(dirname "$0")"/ci_utils.sh

echo "nproc = $(getconf _NPROCESSORS_ONLN) NPROC = ${NPROC}"

install_python_dependencies with-unit-test purge-cache

build_all

df -h

# Run on GPU only. CPU versions run on Github already
if nvidia-smi >/dev/null 2>&1; then
    echo "Try importing cloudViewer Python package"
    test_wheel lib/python_package/pip_package/cloudViewer*.whl
    df -h
    # echo "Running cloudViewer Python tests..."
    # run_python_tests
    # echo
fi