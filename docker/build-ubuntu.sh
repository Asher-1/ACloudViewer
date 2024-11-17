#!/bin/bash

# test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$CLOUDVIEWER_VERSION" && CLOUDVIEWER_VERSION="develop"
test -z "$VTK_VERSION" && VTK_VERSION=9.3.1
test -z "$PCL_VERSION" && PCL_VERSION=1.14.1
test -z "$CUDA_VERSION" && CUDA_VERSION=11.8.0-cudnn8
test -z "$UBUNTU_VERSION" && UBUNTU_VERSION=20.04

BUILD_WITH_CONDA=${BUILD_WITH_CONDA:-OFF}

echo "VTK_VERSION: ${VTK_VERSION}"
echo "PCL_VERSION: ${PCL_VERSION}"
echo "CUDA_VERSION: ${CUDA_VERSION}"
echo "UBUNTU_VERSION: ${UBUNTU_VERSION}"
echo "BUILD_WITH_CONDA: ${BUILD_WITH_CONDA}"
echo "CLOUDVIEWER_VERSION: ${CLOUDVIEWER_VERSION}"

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
	test -f docker_files/qt-opensource-linux-x64-5.14.2.run || \
		wget https://download.qt.io/archive/qt/5.14/5.14.2/qt-opensource-linux-x64-5.14.2.run -O "docker_files/qt-opensource-linux-x64-5.14.2.run"

	test -f docker_files/laszip-src-3.4.3.tar.gz || \
		wget https://raw.githubusercontent.com/Asher-1/CloudViewerUpdate/main/tools/laszip-src-3.4.3.tar.gz -O "docker_files/laszip-src-3.4.3.tar.gz"

	test -f docker_files/xerces-c-3.2.3.zip || \
		wget https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.3.zip -O "docker_files/xerces-c-3.2.3.zip"

	test -f docker_files/metslib-0.5.3.tgz || \
		wget https://www.coin-or.org/download/source/metslib/metslib-0.5.3.tgz -O "docker_files/metslib-0.5.3.tgz"

	test -f docker_files/VTK-${VTK_VERSION}.tar.gz || \
		wget https://vtk.org/files/release/9.3/VTK-${VTK_VERSION}.tar.gz -O "docker_files/VTK-${VTK_VERSION}.tar.gz"

	test -f docker_files/pcl-${PCL_VERSION}.zip || \
		wget https://github.com/PointCloudLibrary/pcl/releases/download/pcl-${PCL_VERSION}/source.zip -O "docker_files/pcl-${PCL_VERSION}.zip"
else
	DOCKER_FILE_POSFIX="_conda"
	BUILD_IMAGE_NAME="cloudviewer-conda"
	DEPENDENCY_IMAGE_NAME="cloudviewer-deps-conda"
fi

# install path
export DOCKER_INSTALL_PATH=/root/install
export HOST_INSTALL_PATH=$PWD/docker_cache/ubuntu$UBUNTU_VERSION${DOCKER_FILE_POSFIX}

# ACloudViewer IMAGE
export CLOUDVIEWER_IMAGE_TAG=${BUILD_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}
# DEPENDENCIES IMAGE
export DEPENDENCY_IMAGE_TAG=${DEPENDENCY_IMAGE_NAME}:${CLOUDVIEWER_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}

if [[ "$(docker images -q $DEPENDENCY_IMAGE_TAG 2> /dev/null)" == "" ]]; 
	then
		docker build \
			--network host \
			--build-arg ALL_PROXY=socks5://127.0.0.1:7890 \
			--build-arg HTTP_PROXY=http://127.0.0.1:7890 \
			--build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
			--build-arg CUDA_VERSION="${CUDA_VERSION}" \
			--build-arg UBUNTU_VERSION="${UBUNTU_VERSION}" \
			--build-arg VTK_VERSION="${VTK_VERSION}" \
			--build-arg PCL_VERSION="${PCL_VERSION}" \
			--tag "$DEPENDENCY_IMAGE_TAG" \
			-f docker/Dockerfile_deps${DOCKER_FILE_POSFIX} . 2>&1 | tee docker_build-${DEPENDENCY_IMAGE_NAME}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}.log
fi

gui_release_export_env() {
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=OFF
    export PACKAGE=ON
}

wheel_release_export_env() {
    export BUILD_SHARED_LIBS=OFF
    export BUILD_CUDA_MODULE=ON
    export BUILD_TENSORFLOW_OPS=OFF
    export BUILD_PYTORCH_OPS=ON
    export PACKAGE=OFF
}

release_build() {
    options="$(echo "$@" | tr ' ' '|')"
    echo "[release_build()] options: ${options}"
    if [[ "py38" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.8
    elif [[ "py39" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.9
    elif [[ "py310" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.10
    elif [[ "py311" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.11
    elif [[ "py312" =~ ^($options)$ ]]; then
        PYTHON_VERSION=3.12
    else
        echo "Invalid python version."
    fi
		if [[ "gui" =~ ^($options)$ ]]; then
        BUILD_GUI=ON
				POST_SUFFIX=GUI
    else
        BUILD_GUI=OFF
				POST_SUFFIX=WHELL
    fi
		if [[ "wheel" =~ ^($options)$ ]]; then
        BUILD_WHEEL=ON
    else
        BUILD_WHEEL=OFF
    fi

		echo "Start building release with python${PYTHON_VERSION}..."
    echo "[release_build()] BUILD_GUI: ${BUILD_GUI}"
    echo "[release_build()] BUILD_WHEEL: ${BUILD_WHEEL}"
    echo "[release_build()] PYTHON_VERSION: ${PYTHON_VERSION}"
    echo "[release_build()] BUILD_TENSORFLOW_OPS=${BUILD_TENSORFLOW_OPS:?'env var must be set.'}"
    echo "[release_build()] BUILD_PYTORCH_OPS=${BUILD_PYTORCH_OPS:?'env var must be set.'}"

    docker build \
			--network host \
			--build-arg ALL_PROXY=socks5://127.0.0.1:7890 \
			--build-arg HTTP_PROXY=http://127.0.0.1:7890 \
			--build-arg HTTPS_PROXY=http://127.0.0.1:7890 \
			--build-arg "CLOUDVIEWER_VERSION=${CLOUDVIEWER_VERSION}" \
			--build-arg "CUDA_VERSION=${CUDA_VERSION}" \
			--build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
			--build-arg "BUILD_GUI=${BUILD_GUI}" \
			--build-arg "BUILD_WHEEL=${BUILD_WHEEL}" \
			--build-arg "DEPENDENCY_IMAGE_NAME=${DEPENDENCY_IMAGE_NAME}" \
			--build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
			--build-arg BUILD_TENSORFLOW_OPS="${BUILD_TENSORFLOW_OPS}" \
			--build-arg BUILD_PYTORCH_OPS="${BUILD_PYTORCH_OPS}" \
			--build-arg BUILD_SHARED_LIBS="${BUILD_SHARED_LIBS}" \
			--build-arg BUILD_CUDA_MODULE="${BUILD_CUDA_MODULE}" \
			--build-arg PACKAGE="${PACKAGE}" \
			--tag "$CLOUDVIEWER_IMAGE_TAG" \
			-f docker/Dockerfile_build${DOCKER_FILE_POSFIX} . 2>&1 | tee docker_build-${PYTHON_VERSION}-${BUILD_IMAGE_NAME}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}-${POST_SUFFIX}.log

			if [ "$BUILD_GUI" = "ON" ]; then
				docker run -v "${HOST_INSTALL_PATH}:/opt/mount" --rm "$CLOUDVIEWER_IMAGE_TAG" \
						bash -cx "cp ${DOCKER_INSTALL_PATH}/*.run /opt/mount \
									&& chown $(id -u):$(id -g) /opt/mount/*.run"
    	fi
			if [ "$BUILD_WHEEL" = "ON" ]; then
				docker run -v "${HOST_INSTALL_PATH}:/opt/mount" --rm "$CLOUDVIEWER_IMAGE_TAG" \
						bash -cx "cp ${DOCKER_INSTALL_PATH}/*.whl /opt/mount \
									&& chown $(id -u):$(id -g) /opt/mount/*.whl"
			fi

			echo					
			echo "Build $CLOUDVIEWER_IMAGE_TAG Done."
			echo "Building ouput package dir is: $HOST_INSTALL_PATH"

			echo
			echo "Start to Delete docker image: $CLOUDVIEWER_IMAGE_TAG."
			docker rmi $CLOUDVIEWER_IMAGE_TAG
			echo "Delete done."
}

# ACloudViewer IMAGE
mkdir -p $HOST_INSTALL_PATH
if [[ "$(docker images -q $CLOUDVIEWER_IMAGE_TAG 2> /dev/null)" == "" ]]; then
	# Start building...
	if ! find "$HOST_INSTALL_PATH" -maxdepth 1 -name "ACloudViewer-*${UBUNTU_VERSION}*.run" | grep -q .; then
    gui_release_export_env
		release_build py38 gui
	else
		echo "Ignore ACloudViewer GUI app building due to have builded before..."
	fi

	if ! find "$HOST_INSTALL_PATH" -maxdepth 1 -name "cloudViewer*-cp38-*.whl" | grep -q .; then
    wheel_release_export_env
		release_build py38 wheel
	else
		echo "Ignore cloudViewer wheel building based on python3.8 due to have builded before..."
	fi

	if ! find "$HOST_INSTALL_PATH" -maxdepth 1 -name "cloudViewer*-cp39-*.whl" | grep -q .; then
    wheel_release_export_env
		release_build py39 wheel
	else
		echo "Ignore cloudViewer wheel building based on python3.9 due to have builded before..."
	fi

	if ! find "$HOST_INSTALL_PATH" -maxdepth 1 -name "cloudViewer*-cp310-*.whl" | grep -q .; then
    wheel_release_export_env
		release_build py310 wheel
	else
		echo "Ignore cloudViewer wheel building based on python3.10 due to have builded before..."
	fi

	if ! find "$HOST_INSTALL_PATH" -maxdepth 1 -name "cloudViewer*-cp311-*.whl" | grep -q .; then
    wheel_release_export_env
		release_build py311 wheel
	else
		echo "Ignore cloudViewer wheel building based on python3.11 due to have builded before..."
	fi

	if [ "$UBUNTU_VERSION" != "18.04" ]; then
		if ! find "$HOST_INSTALL_PATH" -maxdepth 1 -name "cloudViewer*-cp312-*.whl" | grep -q .; then
			wheel_release_export_env
			release_build py312 wheel
		else
			echo "Ignore cloudViewer wheel building based on python3.12 due to have builded before..."
		fi
	else
		echo "Ubuntu18.04 do not support python3.12 as default!"
	fi
else
	echo "Please run docker rmi $CLOUDVIEWER_IMAGE_TAG first!"
fi

