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

echo "Start to build GUI package On MacOS..."
echo
build_gui_app with_conda package_installer

df -h