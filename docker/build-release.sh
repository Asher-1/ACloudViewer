#!/bin/bash

set -euo pipefail

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

# build ubuntu 20.04 wheel
CUDA_VERSION=12.6.3-cudnn UBUNTU_VERSION=20.04 docker/docker_build_local.sh
