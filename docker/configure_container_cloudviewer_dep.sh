#!/bin/bash

# install some dependence on host pc
sudo apt-get install x11-xserver-utils && xhost +
# ssh��ͨ�û���¼����
# ssh -p 10022 ubuntu@127.0.0.1
export DISPLAY=10.147.17.208:0.0
# export DISPLAY=:0

# create container instance
docker run -dit --runtime=nvidia --name=cloudviewer \
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
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/build:/opt/ACloudViewer/build \
  -v /home/asher/develop/code/github/CloudViewer/ACloudViewer/docker_cache/install:/opt/ACloudViewer/install \
  -v /home/asher/develop/thirdparty:/opt/ACloudViewer/thirdparties \
  -v /home/asher/develop/code/github/CloudViewer/CloudViewer-ML:/opt/ACloudViewer/CloudViewer-ML \
  cloudviewer-deps:develop-ubuntu18.04-cuda101


# attach into container instance
docker exec -it cloudviewer /bin/bash

ln -s /opt/Qt5.14.2/5.14.2/gcc_64/lib/libQt5X11Extras.so.5.14.2 /usr/lib/libQt5X11Extras.so.5
export PATH="/opt/Qt5.14.2/5.14.2/gcc_64/bin:$PATH"
export LD_LIBRARY_PATH="/opt/Qt5.14.2/5.14.2/gcc_64/lib:$LD_LIBRARY_PATH"
export QT_PLUGIN_PATH="/opt/Qt5.14.2/5.14.2/gcc_64/plugins:$QT_PLUGIN_PATH"
export QML2_IMPORT_PATH="/opt/Qt5.14.2/5.14.2/gcc_64/qml:$QML2_IMPORT_PATH"

# Build ACloudViewer
export DISPLAY=:0  # DISPLAY must be consistent with host
cd /opt/ACloudViewer
if [ ! -d "build" ]; then # dir does not exist
  echo "creating dir build..."
  mkdir build
fi
cd build

cmakeWhlOptions=(-DDEVELOPER_BUILD=OFF
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_JUPYTER_EXTENSION=ON
  -DBUILD_LIBREALSENSE=ON
  -DBUILD_AZURE_KINECT=ON
  -DBUILD_RPC_INTERFACE=ON
  -DBUILD_BENCHMARKS=OFF
  -DWITH_OPENMP=ON
  -DBUILD_CUDA_MODULE=ON
  -DBUILD_PYTORCH_OPS=ON
  -DBUILD_TENSORFLOW_OPS=OFF
  -DBUNDLE_CLOUDVIEWER_ML=ON
  -DCLOUDVIEWER_ML_ROOT=/opt/ACloudViewer/CloudViewer-ML
  -DTHIRD_PARTY_DOWNLOAD_DIR=/opt/ACloudViewer/thirdparties
  -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python3.6
  -DPYTHON_IN_PATH=/root/miniconda3/bin/python3.6
  -DPYTHON_LIBRARY=/root/miniconda3/lib/libpython3.6m.so
  -DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake
  -DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake
)
cmake "/opt/ACloudViewer/ACloudViewer" \
      "${cmakeWhlOptions[@]}" \
      -DCMAKE_INSTALL_PREFIX=/opt/ACloudViewer/install

# compile ACloudViewer pip-package
make "-j$(nproc)" pip-package
make "-j$(nproc)" conda-package
make install "-j$(nproc)"

cmakeGuiOptions=(-DDEVELOPER_BUILD=OFF
                -DCMAKE_BUILD_TYPE=Release
                -DBUILD_JUPYTER_EXTENSION=ON
                -DBUILD_LIBREALSENSE=ON
                -DBUILD_AZURE_KINECT=ON
                -DBUILD_RPC_INTERFACE=ON
                -DBUILD_BENCHMARKS=OFF
                -DBUILD_OPENCV=ON
                -DWITH_OPENMP=ON
                -DBUILD_CUDA_MODULE=OFF
                -DBUILD_PYTORCH_OPS=OFF
                -DBUILD_TENSORFLOW_OPS=OFF
                -DBUNDLE_CLOUDVIEWER_ML=OFF
                -DGLIBCXX_USE_CXX11_ABI=ON
                -DCVCORELIB_USE_CGAL=ON
                -DCVCORELIB_SHARED=ON
                -DCVCORELIB_USE_QT_CONCURRENT=ON
                -DOPTION_USE_DXF_LIB=ON
                -DOPTION_USE_RANSAC_LIB=ON
                -DOPTION_USE_SHAPE_LIB=ON
                -DPLUGIN_IO_QADDITIONAL=ON
                -DPLUGIN_IO_QCORE=ON
                -DPLUGIN_IO_QCSV_MATRIX=ON
                -DPLUGIN_IO_QE57=ON
                -DPLUGIN_IO_QMESH=ON
                -DPLUGIN_IO_QPDAL=OFF
                -DPLUGIN_IO_QPHOTOSCAN=ON
                -DPLUGIN_IO_QRDB=ON
                -DPLUGIN_IO_QRDB_FETCH_DEPENDENCY=ON
                -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON
                -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON
                -DPLUGIN_STANDARD_QANIMATION=ON
                -DPLUGIN_STANDARD_QCANUPO=ON
                -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON
                -DPLUGIN_STANDARD_QCOMPASS=ON
                -DPLUGIN_STANDARD_QCSF=ON
                -DPLUGIN_STANDARD_QFACETS=ON
                -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON
                -DPLUGIN_STANDARD_QM3C2=ON
                -DPLUGIN_STANDARD_QMPLANE=ON
                -DPLUGIN_STANDARD_QPCL=ON
                -DPLUGIN_STANDARD_QPOISSON_RECON=ON
                -DPOISSON_RECON_WITH_OPEN_MP=ON
                -DPLUGIN_STANDARD_QRANSAC_SD=ON
                -DPLUGIN_STANDARD_QSRA=ON
                -DTHIRD_PARTY_DOWNLOAD_DIR=/opt/ACloudViewer/thirdparties
                -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python3.6
                -DPYTHON_IN_PATH=/root/miniconda3/bin/python3.6
                -DPYTHON_LIBRARY=/root/miniconda3/lib/libpython3.6m.so
                -DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake
                -DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake
        )

cmake "/opt/ACloudViewer/ACloudViewer" \
      "${cmakeGuiOptions[@]}" \
      -DCMAKE_INSTALL_PREFIX=/opt/ACloudViewer/install

make "-j$(nproc)"
make install "-j$(nproc)"

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