#!/bin/bash

# install some dependence on host pc
# sudo apt-get install x11-xserver-utils && xhost +
# ssh -p 10022 ubuntu@127.0.0.1
# export DISPLAY=10.147.17.208:0
export DISPLAY=:0

docker run -dit --name=test_cloudviewer_dep_conda \
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
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/install:/root/install \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/build:/root/ACloudViewer/build \
  cloudviewer-deps-conda:develop-ubuntu18.04-cuda11.8.0-cudnn8

# attach into container instance
docker exec -it test_cloudviewer_dep_conda /bin/bash

# create container instance
docker run -dit --name=test_cloudviewer_conda \
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
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/install:/root/install \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/build:/root/ACloudViewer/build \
  cloudviewer-conda:develop-ubuntu18.04-cuda11.8.0-cudnn8


# attach into container instance
docker exec -it test_cloudviewer_conda /bin/bash

export ACloudViewer_DEV=/root \
export ACloudViewer_BUILD=/root/ACloudViewer/build \
export ACloudViewer_INSTALL=/root/install \
export BUNDLE_CLOUDVIEWER_ML=/root/CloudViewer-ML \
export PATH="/root/miniconda3/bin:$PATH:$BUNDLE_CLOUDVIEWER_ML" \

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# build ACloudViewer app installer
rm -rf ${ACloudViewer_BUILD}/* && ./docker/build_gui_app_conda.sh 3.12 OFF
rm -rf ${ACloudViewer_BUILD}/* && ./docker/build_cloudviewer_whl_conda.sh 3.12

test cloudViewer
python3 -c "import cloudViewer as cv3d; print(cv3d.__version__); print('CUDA available: ', cv3d.core.cuda.is_available());"

