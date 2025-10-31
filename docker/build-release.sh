#!/bin/bash

set -euo pipefail

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

# build ubuntu 20.04 wheel
# due to some crash issues with Reconstruction module on ubuntu 22.04
CUDA_VERSION=12.6.3-cudnn UBUNTU_VERSION=20.04 docker/build-ubuntu.sh
