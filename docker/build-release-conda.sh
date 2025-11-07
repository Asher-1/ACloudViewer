#!/bin/bash

set -euo pipefail

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

# building with conda on ubuntu18.04 issues: '(1.0e+0 / 3.0e+0)' is not a constant expression
BUILD_WITH_CONDA=ON CUDA_VERSION=12.6.3-cudnn UBUNTU_VERSION=22.04 docker/docker_build_local.sh
