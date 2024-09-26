#!/bin/bash
set -e

#test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="develop"
test -z "$VTK_VERSION" && VTK_VERSION=8.2.0
test -z "$PCL_VERSION" && PCL_VERSION=1.11.1
test -z "$CUDA_VERSION" && CUDA_VERSION=11.8
test -z "$UBUNTU_VERSION" && UBUNTU_VERSION=18.04

test -d docker || (
        echo This script must be run from the top level ACloudViewer directory
	exit 1
)

test -d docker_files || \
	mkdir docker_files

test -f docker_files/qt.run || \
	wget https://download.qt.io/archive/qt/5.14/5.14.2/qt-opensource-linux-x64-5.14.2.run -O "docker_files/qt.run"

test -f docker_files/Miniconda3-latest-Linux-x86_64.sh || \
	wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "docker_files/Miniconda3-latest-Linux-x86_64.sh"

test -f docker_files/xerces-c-3.2.3.zip || \
	wget https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.3.zip -O "docker_files/xerces-c-3.2.3.zip"

test -f docker_files/VTK-9.3.1.tar.gz || \
	wget https://vtk.org/files/release/9.3/VTK-9.3.1.tar.gz -O "docker_files/VTK-9.3.1.tar.gz"

test -f docker_files/pcl-1.11.1.zip || \
	wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-1.11.1/source.zip -O "docker_files/pcl-1.11.1.zip"

test -f docker_files/nomachine.deb || \
	wget "https://www.nomachine.com/free/linux/64/deb" -O "docker_files/nomachine.deb"

test -f docker_files/QtIFW-4.6.1-linux-amd.zip || \
	wget "https://raw.githubusercontent.com/Asher-1/CloudViewerUpdate/main/tools/QtIFW-4.6.1-linux-amd.zip" -O "docker_files/QtIFW-4.6.1-linux-amd.zip"
	

if [[ "$(docker images -q cloudviewer-deps:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION} 2> /dev/null)" == "" ]]; 
	then
		# DEPENDENCIES
		docker build \
			--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
			--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
			--build-arg "VTK_VERSION=${VTK_VERSION}" \
			--build-arg "PCL_VERSION=${PCL_VERSION}" \
			--tag "cloudviewer-deps:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
			-f docker/Dockerfile_ubuntu_deps . --progress=plain 2>&1 | tee docker_build-cloudviewer-deps-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}.log
fi

# ACloudViewer
docker build \
	--build-arg "CLOUDVIEWER_VERSION=${CLOUDVIEWER_VERSION}" \
	--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
	--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
	--build-arg "VTK_VERSION=${VTK_VERSION}" \
	--build-arg "PCL_VERSION=${PCL_VERSION}" \
	--tag "cloudviewer:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
	-f docker/Dockerfile_ubuntu . --progress=plain 2>&1 | tee docker_build-cloudviewer-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}.log

# Export docker compiling output data
docker_install_package_dir=/root/install
host_install_package_dir=$PWD/docker_cache/ubuntu$UBUNTU_VERSION
mkdir -p $host_install_package_dir
docker run -v "${host_install_package_dir}:/opt/mount" --rm cloudviewer:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION} \
    bash -c "cp ${docker_install_package_dir}/*.whl /opt/mount \
          && cp ${docker_install_package_dir}/*.run /opt/mount \
          && chown $(id -u):$(id -g) /opt/mount/*.whl \
          && chown $(id -u):$(id -g) /opt/mount/*.run"
echo					
echo "Build cloudviewer:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION} Done."
echo "Building ouput package dir is: $host_install_package_dir"
# echo "Start to Delete docker image: cloudviewer:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}."
# docker rmi cloudviewer:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}
# echo "Delete done."
