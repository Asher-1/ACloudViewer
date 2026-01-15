#!/bin/bash

# install some dependence on host pc
# sudo apt-get install x11-xserver-utils && xhost +
# ssh -p 10022 ubuntu@127.0.0.1
# export DISPLAY=10.147.17.208:0.0
# export DISPLAY=:0

# create container instance
docker run -dit --name=test_cloudviewer_dep_ubuntu2004 \
  --shm-size="16g" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --net=host \
  --ipc=host \
  --gpus=all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env ALL_PROXY=socks5://127.0.0.1:7890 \
	--env HTTP_PROXY=http://127.0.0.1:7890 \
	--env HTTPS_PROXY=http://127.0.0.1:7890 \
  --env PIP_DEFAULT_TIMEOUT=1000 \
  --env PIP_RETRIES=5 \
  --env PIP_TIMEOUT=1000 \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  -p 10022:22 \
  -p 14000:4000 \
  -e "QT_X11_NO_MITSHM=1" \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer:/root/ACloudViewer \
  -v /home/asher/develop/code/github/CloudViewer/CloudViewer-ML:/root/CloudViewer-ML \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/install:/root/install \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/build:/root/ACloudViewer/build \
  cloudviewer-deps:develop-ubuntu20.04-cuda12.6.3-cudnn

docker exec -it test_cloudviewer_dep_ubuntu2004 /bin/bash

docker run -dit --name=test_cloudviewer_dep_ubuntu2204 \
  --shm-size="16g" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --net=host \
  --ipc=host \
  --gpus=all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env ALL_PROXY=socks5://127.0.0.1:7890 \
	--env HTTP_PROXY=http://127.0.0.1:7890 \
	--env HTTPS_PROXY=http://127.0.0.1:7890 \
  --env PIP_DEFAULT_TIMEOUT=1000 \
  --env PIP_RETRIES=5 \
  --env PIP_TIMEOUT=1000 \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  -p 10022:22 \
  -p 14000:4000 \
  -e "QT_X11_NO_MITSHM=1" \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer:/root/ACloudViewer \
  -v /home/asher/develop/code/github/CloudViewer/CloudViewer-ML:/root/CloudViewer-ML \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/install:/root/install \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/build:/root/ACloudViewer/build \
  cloudviewer-deps:develop-ubuntu22.04-cuda12.6.3-cudnn

# attach into container instance
docker exec -it test_cloudviewer_dep_ubuntu2204 /bin/bash

docker run -dit --name=test_cloudviewer \
  --shm-size="16g" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --net=host \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  --env ALL_PROXY=socks5://127.0.0.1:7890 \
	--env HTTP_PROXY=http://127.0.0.1:7890 \
	--env HTTPS_PROXY=http://127.0.0.1:7890 \
  --env PIP_DEFAULT_TIMEOUT=1000 \
  --env PIP_RETRIES=5 \
  --env PIP_TIMEOUT=1000 \
  --ipc=host \
  --gpus=all \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  -p 10022:22 \
  -p 14000:4000 \
  -e "QT_X11_NO_MITSHM=1" \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer:/root/ACloudViewer \
  -v /home/asher/develop/code/github/CloudViewer/CloudViewer-ML:/root/CloudViewer-ML \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/install:/root/install \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/build:/root/ACloudViewer/build \
  cloudviewer:develop-ubuntu18.04-cuda12.6.3-cudnn


# attach into container instance
docker exec -it test_cloudviewer /bin/bash

cd /root/ACloudViewer
export QT_BASE_DIR=/usr/lib/x86_64-linux-gnu/qt5
export ACloudViewer_DEV=/root
export ACloudViewer_BUILD=/root/ACloudViewer/build
export ACloudViewer_INSTALL=/root/install
export CLOUDVIEWER_ML_ROOT=/root/CloudViewer-ML
export LD_LIBRARY_PATH=${QT_BASE_DIR}/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=${QT_BASE_DIR}/lib/pkgconfig:$PKG_CONFIG_PATH

# create python env
export PYTHON_VERSION=3.12
export PIP_DEFAULT_TIMEOUT=1000
export PIP_RETRIES=5
export PIP_TIMEOUT=1000
export PYENV_ROOT=/root/.pyenv
export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PYENV_ROOT/versions/$PYTHON_VERSION/bin:$PATH";

curl https://pyenv.run | bash \
        && pyenv update \
        && pyenv install $PYTHON_VERSION \
        && pyenv global $PYTHON_VERSION \
        && pyenv rehash \
        && ln -s $PYENV_ROOT/versions/${PYTHON_VERSION}* $PYENV_ROOT/versions/${PYTHON_VERSION}
python --version && pip --version

./docker/scripts/configure_pip_timeout.sh

# build ACloudViewer app installer
rm -rf ${ACloudViewer_BUILD}/* && ./docker/build_gui_app.sh $PYTHON_VERSION ON

git config --global --add safe.directory /root/CloudViewer-ML/.git
rm -rf ${ACloudViewer_BUILD}/* && ./docker/build_cloudviewer_whl.sh $PYTHON_VERSION

test cloudViewer
python3 -c "import cloudViewer as cv3d; print(cv3d.__version__); print('CUDA available: ', cv3d.core.cuda.is_available());"
python3 -c "import cloudViewer.ml.torch as ml3d"


debug ACloudViewer App
export LD_LIBRARY_PATH=~/ACloudViewer/lib
export PYTHONPATH=~/ACloudViewer/plugins/Python
cd ~/ACloudViewer
gdb ./ACloudViewer
run
bt


pcl: metslib-0.5.3
wget https://www.coin-or.org/download/source/metslib/metslib-0.5.3.tgz
tar xjvf metslib-0.5.3.tgz
cd metslib-0.5.3
sh ./configure
make
sudo make install

pcl: dependency libusb-1.0:
(sudo apt-get install libkmod-dev libblkid-dev libglib2.0-dev)(optional)

pcl+RealSense: libudev-dev
git clone https://github.com/illiliti/libudev-zero.git
cd libudev-zero
make
sudo make PREFIX=/usr install

wget http://downloads.sourceforge.net/project/libusb/libusb-1.0/libusb-1.0.23/libusb-1.0.23.tar.bz2
tar xjvf libusb-1.0.23.tar.bz2
cd libusb-1.0.23
./configure
make
sudo make install
