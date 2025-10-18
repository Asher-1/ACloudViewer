#!/bin/bash

set -euo pipefail

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

CUDA_VERSION=12.6.3-cudnn UBUNTU_VERSION=22.04 docker/build-ubuntu.sh
