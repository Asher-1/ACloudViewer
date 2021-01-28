#!/bin/bash
set -e

#test -z "$EROWCLOUDVIEWER_VERSION" && EROWCLOUDVIEWER_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$EROWCLOUDVIEWER_VERSION" && EROWCLOUDVIEWER_VERSION="develop"
test -z "$VTK_VERSION" && VTK_VERSION=8.2.0
test -z "$PCL_VERSION" && PCL_VERSION=1.11.1
test -z "$CUDA_VERSION" && CUDA_VERSION=10.2
test -z "$UBUNTU_VERSION" && UBUNTU_VERSION=18.04

test -d docker || (
        echo This script must be run from the top level ErowCloudViewer directory
	exit 1
)

test -d dl || \
	mkdir dl
test -f dl/qt.run || \
	wget https://download.qt.io/archive/qt/5.14/5.14.2/qt-opensource-linux-x64-5.14.2.run -O "dl/qt.run"

# DEPENDENCIES
#docker build \
#	--rm \
#	--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
#	--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
#	--build-arg "VTK_VERSION=${VTK_VERSION}" \
#	--build-arg "PCL_VERSION=${PCL_VERSION}" \
#	--tag "erowcloudviewer-deps:${EROWCLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
#	-f docker/Dockerfile_ubuntu_deps .

# ErowCloudViewer
docker build \
	--rm \
	--build-arg "EROWCLOUDVIEWER_VERSION=${EROWCLOUDVIEWER_VERSION}" \
	--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
	--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
  --build-arg "VTK_VERSION=${VTK_VERSION}" \
	--build-arg "PCL_VERSION=${PCL_VERSION}" \
	--tag "erowcloudviewer:${EROWCLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
	-f docker/Dockerfile_ubuntu .

