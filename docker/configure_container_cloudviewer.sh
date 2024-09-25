#!/bin/bash

# install some dependence on host pc
# sudo apt-get install x11-xserver-utils && xhost +
# ssh -p 10022 ubuntu@127.0.0.1
# export DISPLAY=10.147.17.208:0
export DISPLAY=:0

# create container instance
docker run -dit --name=test_cloudviewer \
  --shm-size="16g" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --net=host \
  --ipc=host \
  --gpus=all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -e DISPLAY=unix$DISPLAY \
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
  registry.cn-shanghai.aliyuncs.com/asher-ai/cloudviewer:develop-ubuntu18.04-cuda101


# attach into container instance
docker exec -it test_cloudviewer /bin/bash

ENV DBUS_SYSTEM_BUS_ADDRESS=unix:path=/host/run/dbus/system_bus_socket \
    USER=ubuntu \
    PASSWD=ubuntu \
    UID=1000 \
    GID=1000 \
    TZ=Asia/Shanghai \
    LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    LANGUAGE=zh_CN.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

RUN groupadd -f $USER \
    && useradd --create-home --no-log-init -g $USER $USER \
    && usermod -aG sudo $USER \
    && echo "$USER:$PASSWD" | chpasswd \
    && chsh -s /bin/bash $USER \
    && usermod  --uid $UID $USER \
    && groupmod --gid $GID $USER

# Install some dependences and xfce4 desktop
RUN apt-get update --fix-missing -y \
    && apt install  --fix-missing -yq \
    openssh-server \
    bash-completion \
    xfce4 \
    xfce4-terminal \
    xfce4-power-manager \
    fonts-wqy-zenhei \
    locales \
    ssh xauth \
	&& systemctl enable ssh \
	&& mkdir -p /run/sshd \
	&& locale-gen $LANG \
	&& /bin/sh -c LANG=C xdg-user-dirs-update --force

COPY docker_files/google-chrome-stable_current_amd64.deb /opt
COPY docker_files/nomachine.deb /opt
RUN apt-get install -yf ./google-chrome-stable_current_amd64.deb \
    && rm ./google-chrome-stable_current_amd64.deb \
    && apt-get install -y pulseaudio \
    && mkdir -p /var/run/dbus \
    && dpkg -i ./nomachine.deb \
    && sed -i "s|#EnableClipboard both|EnableClipboard both |g" /usr/NX/etc/server.cfg \
    && sed -i '/DefaultDesktopCommand/c\DefaultDesktopCommand "/usr/bin/startxfce4"' /usr/NX/etc/node.cfg