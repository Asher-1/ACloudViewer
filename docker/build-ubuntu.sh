#!/bin/bash

# test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="develop"
test -z "$VTK_VERSION" && VTK_VERSION=9.3.1
test -z "$PCL_VERSION" && PCL_VERSION=1.14.1
test -z "$CUDA_VERSION" && CUDA_VERSION=11.7.1-cudnn8
test -z "$UBUNTU_VERSION" && UBUNTU_VERSION=18.04

BUILD_WITH_CONDA=${BUILD_WITH_CONDA:-OFF}

# put after test -z due to unbound issues
set -euo pipefail
export BUILDKIT_PROGRESS=plain

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

test -d docker_files || \
	mkdir docker_files


test -f docker_files/Miniconda3-latest-Linux-x86_64.sh || \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "docker_files/Miniconda3-latest-Linux-x86_64.sh"

test -f docker_files/QtIFW-4.6.1-linux-amd.zip || \
	wget "https://raw.githubusercontent.com/Asher-1/CloudViewerUpdate/main/tools/QtIFW-4.6.1-linux-amd.zip" -O "docker_files/QtIFW-4.6.1-linux-amd.zip"

if [ "$BUILD_WITH_CONDA" != "ON" ]; then
	DOCKER_FILE_POSFIX=""
	BUILD_IMAGE_NAME="cloudviewer"
	DEPENDENCY_IMAGE_NAME="cloudviewer-deps"
	test -f docker_files/qt.run || \
		wget https://download.qt.io/archive/qt/5.14/5.14.2/qt-opensource-linux-x64-5.14.2.run -O "docker_files/qt.run"

	test -f docker_files/laszip-src-3.4.3.tar.gz || \
		wget ttps://raw.githubusercontent.com/Asher-1/CloudViewerUpdate/main/tools/laszip-src-3.4.3.tar.gz -O "docker_files/laszip-src-3.4.3.tar.gz"

	test -f docker_files/xerces-c-3.2.3.zip || \
		wget https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.3.zip -O "docker_files/xerces-c-3.2.3.zip"

	test -f docker_files/metslib-0.5.3.tgz || \
		wget https://www.coin-or.org/download/source/metslib/metslib-0.5.3.tgz -O "docker_files/metslib-0.5.3.tgz"

	test -f docker_files/VTK-9.3.1.tar.gz || \
		wget https://vtk.org/files/release/9.3/VTK-9.3.1.tar.gz -O "docker_files/VTK-9.3.1.tar.gz"

	test -f docker_files/pcl-1.14.1.zip || \
		wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.14.1/source.zip -O "docker_files/pcl-1.14.1.zip"
else
	DOCKER_FILE_POSFIX="_conda"
	BUILD_IMAGE_NAME="cloudviewer-conda"
	DEPENDENCY_IMAGE_NAME="cloudviewer-deps-conda"
fi

if [[ "$(docker images -q ${DEPENDENCY_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION} 2> /dev/null)" == "" ]]; 
	then
		# DEPENDENCIES
		docker build \
			--network host \
			--build-arg ALL_PROXY=socks5://127.0.0.1:7890 \
			--build-arg HTTP_PROXY=http://127.0.0.1:7890 \
			--build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
			--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
			--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
			--build-arg "VTK_VERSION=${VTK_VERSION}" \
			--build-arg "PCL_VERSION=${PCL_VERSION}" \
			--tag "${DEPENDENCY_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
			-f docker/Dockerfile_deps${DOCKER_FILE_POSFIX} . 2>&1 | tee docker_build-${DEPENDENCY_IMAGE_NAME}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}.log
fi

# ACloudViewer
if [[ "$(docker images -q ${BUILD_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION} 2> /dev/null)" == "" ]]; 
	then
	# Start building...
	docker build \
		--network host \
		--build-arg ALL_PROXY=socks5://127.0.0.1:7890 \
		--build-arg HTTP_PROXY=http://127.0.0.1:7890 \
		--build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
		--build-arg "CLOUDVIEWER_VERSION=${CLOUDVIEWER_VERSION}" \
		--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
		--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
		--build-arg "VTK_VERSION=${VTK_VERSION}" \
		--build-arg "PCL_VERSION=${PCL_VERSION}" \
		--build-arg "DEPENDENCY_IMAGE_NAME=${DEPENDENCY_IMAGE_NAME}" \
		--tag "${BUILD_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
		-f docker/Dockerfile_build${DOCKER_FILE_POSFIX} . 2>&1 | tee docker_build-${BUILD_IMAGE_NAME}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}.log

	# Export docker compiling output data
	docker_install_package_dir=/root/install
	host_install_package_dir=$PWD/docker_cache/ubuntu$UBUNTU_VERSION${DOCKER_FILE_POSFIX}
	mkdir -p $host_install_package_dir
	docker run -v "${host_install_package_dir}:/opt/mount" --rm ${BUILD_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION} \
			bash -c "cp ${docker_install_package_dir}/*.whl /opt/mount \
						&& cp ${docker_install_package_dir}/*.run /opt/mount \
						&& chown $(id -u):$(id -g) /opt/mount/*.whl \
						&& chown $(id -u):$(id -g) /opt/mount/*.run"
	echo					
	echo "Build ${BUILD_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION} Done."
	echo "Building ouput package dir is: $host_install_package_dir"
fi

echo "Start to Delete docker image: ${BUILD_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}."
docker rmi ${BUILD_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}
echo "Delete done."
