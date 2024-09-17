#!/bin/sh

set -e

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

#CUDA_VERSION=110 UBUNTU_VERSION=20.04 docker/build-ubuntu.sh
#CUDA_VERSION=102 UBUNTU_VERSION=20.04 docker/build-ubuntu.sh
#CUDA_VERSION=110 UBUNTU_VERSION=18.04 docker/build-ubuntu.sh
#CUDA_VERSION=102 UBUNTU_VERSION=18.04 docker/build-ubuntu.sh
#CUDA_VERSION=101 UBUNTU_VERSION=18.04 docker/build-ubuntu.sh
# CUDA_VERSION=101 UBUNTU_VERSION=18.04 docker/build-ubuntu_on_macos.sh
CUDA_VERSION=11.8.0 UBUNTU_VERSION=20.04 docker/build-ubuntu.sh