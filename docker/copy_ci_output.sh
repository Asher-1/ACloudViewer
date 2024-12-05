#!/bin/bash

CI_CONFIG=$1
BUILD_GUI=$2
CLOUDVIEWER_CI_IMAGE_TAG="cloudviewer-ci:$CI_CONFIG"

echo "BUILD_GUI: ${BUILD_GUI}"
echo "CLOUDVIEWER_CI_IMAGE_TAG: ${CLOUDVIEWER_CI_IMAGE_TAG}"

export DOCKER_INSTALL_PATH=/root/install
export HOST_INSTALL_PATH=$PWD/docker_cache/ci_$CI_CONFIG
mkdir -p $HOST_INSTALL_PATH
if [ "$BUILD_GUI" = "ON" ]; then
	docker run -v "${HOST_INSTALL_PATH}:/opt/mount" --rm "$CLOUDVIEWER_CI_IMAGE_TAG" \
			bash -cx "cp ${DOCKER_INSTALL_PATH}/*.run /opt/mount \
						&& chown $(id -u):$(id -g) /opt/mount/*.run"
else
	docker run -v "${HOST_INSTALL_PATH}:/opt/mount" --rm "$CLOUDVIEWER_CI_IMAGE_TAG" \
			bash -cx "cp ${DOCKER_INSTALL_PATH}/*.whl /opt/mount \
						&& chown $(id -u):$(id -g) /opt/mount/*.whl"
fi