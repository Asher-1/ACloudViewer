#!/bin/bash
set -e

#test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="develop"
test -z "$VTK_VERSION" && VTK_VERSION=8.2.0
test -z "$PCL_VERSION" && PCL_VERSION=1.11.1
test -z "$CUDA_VERSION" && CUDA_VERSION=101
test -z "$UBUNTU_VERSION" && UBUNTU_VERSION=18.04

test -d docker || (
        echo This script must be run from the top level ErowCloudViewer directory
	exit 1
)

test -d dl || \
	mkdir dl

test -f dl/qt.run || \
	wget https://download.qt.io/archive/qt/5.14/5.14.2/qt-opensource-linux-x64-5.14.2.run -O "dl/qt.run"

test -f dl/Miniconda3-latest-Linux-x86_64.sh || \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "dl/Miniconda3-latest-Linux-x86_64.sh"

test -f dl/xerces-c-3.2.3.zip || \
	wget https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.3.zip -O "dl/xerces-c-3.2.3.zip"

test -f dl/VTK-8.2.0.zip || \
	wget https://www.vtk.org/files/release/8.2/VTK-8.2.0.zip -O "dl/VTK-8.2.0.zip"

test -f dl/pcl-1.11.1.zip || \
	wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.11.1/source.zip -O "dl/pcl-1.11.1.zip"

test -f dl/opencv.zip || \
	wget https://github.com/opencv/opencv/archive/4.3.0.zip -O "dl/opencv.zip"

test -f dl/opencv_contrib.zip || \
	wget https://github.com/opencv/opencv_contrib/archive/4.3.0.zip -O "dl/opencv_contrib.zip"

test -f dl/google-chrome-stable_current_amd64.deb || \
	wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -O "dl/google-chrome-stable_current_amd64.deb"

test -f dl/nomachine.deb || \
	wget "https://www.nomachine.com/free/linux/64/deb" -O "dl/nomachine.deb"

# DEPENDENCIES
docker build \
	--rm \
	--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
	--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
	--build-arg "VTK_VERSION=${VTK_VERSION}" \
	--build-arg "PCL_VERSION=${PCL_VERSION}" \
	--tag "cloudviewer-deps:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
	-f docker/Dockerfile_ubuntu_deps .

# ErowCloudViewer
#docker build \
#	--rm \
#	--build-arg "CLOUDVIEWER_VERSION=${CLOUDVIEWER_VERSION}" \
#	--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
#	--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
#  --build-arg "VTK_VERSION=${VTK_VERSION}" \
#	--build-arg "PCL_VERSION=${PCL_VERSION}" \
#	--tag "cloudviewer:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
#	-f docker/Dockerfile_ubuntu .

