#!/bin/bash

set -euo pipefail

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

CUDA_VERSION=12.6.3-cudnn UBUNTU_VERSION=24.04 docker/docker_build_local.sh \
&& CUDA_VERSION=12.6.3-cudnn UBUNTU_VERSION=22.04 docker/docker_build_local.sh \
&& CUDA_VERSION=12.6.3-cudnn UBUNTU_VERSION=20.04 docker/docker_build_local.sh \
&& CUDA_VERSION=11.8.0-cudnn8 UBUNTU_VERSION=18.04 docker/docker_build_local.sh