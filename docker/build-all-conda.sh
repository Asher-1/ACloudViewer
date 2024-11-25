#!/bin/bash

set -euo pipefail

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

BUILD_WITH_CONDA=ON CUDA_VERSION=11.8.0-cudnn8 UBUNTU_VERSION=20.04 docker/build-ubuntu.sh    \
&& BUILD_WITH_CONDA=ON CUDA_VERSION=11.8.0-cudnn8 UBUNTU_VERSION=22.04 docker/build-ubuntu.sh
# building with conda on ubuntu18.04 issues: '(1.0e+0 / 3.0e+0)' is not a constant expression
# && BUILD_WITH_CONDA=ON CUDA_VERSION=11.8.0-cudnn8 UBUNTU_VERSION=18.04 docker/build-ubuntu.sh \