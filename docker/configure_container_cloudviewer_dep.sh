#!/bin/bash

# install some dependence on host pc
# sudo apt-get install x11-xserver-utils && xhost +
# ssh -p 10022 ubuntu@127.0.0.1
# export DISPLAY=10.147.17.208:0.0
# export DISPLAY=:0

# create container instance
docker run -dit --name=test_cloudviewer_dep \
  --shm-size="16g" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --net=host \
  --ipc=host \
  --gpus=all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
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
  cloudviewer-deps:develop-ubuntu20.04-cuda11.8.0

# attach into container instance
docker exec -it test_cloudviewer_dep /bin/bash

cd /root/ACloudViewer
export ACloudViewer_DEV=/root \
export ACloudViewer_BUILD=/root/ACloudViewer/build \
export ACloudViewer_INSTALL=/root/install \
export BUNDLE_CLOUDVIEWER_ML=/root/CloudViewer-ML \
export QT_DIR=/opt/Qt5.14.2/5.14.2/gcc_64 \
export PATH="/root/miniconda3/bin:$PATH:$BUNDLE_CLOUDVIEWER_ML" \
export LD_LIBRARY_PATH="/opt/Qt5.14.2/5.14.2/gcc_64/lib:$LD_LIBRARY_PATH"
rm -rf ${ACloudViewer_BUILD}/* && ./docker/build_cloudviewer_whl.sh 3.10

# build ACloudViewer app installer
rm -rf ${ACloudViewer_BUILD}/* && ./docker/build_gui_app.sh