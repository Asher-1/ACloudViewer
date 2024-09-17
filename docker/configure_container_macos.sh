#!/bin/bash

IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
xhost + $IP
DISPLAY=:0

# ssh��ͨ�û���¼����
# ssh -p 20022 ubuntu@127.0.0.1
# export DISPLAY=192.168.1.5:0
# brew install xquartz socat
# socat TCP-LISTEN:6000,reuseaddr,fork UNIX-CLIENT:\"$DISPLAY\"
# IP=$(ifconfig en0 | grep inet | awk '$1=="inet" {print $2}')
# xhost + $IP
# export DISPLAY=:0
# startxfce4 &

# create container instance
docker run -dit --name=cloudviewer_env \
  --shm-size="2g" \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  -e DISPLAY=$IP$DISPLAY \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  -p 20022:22 \
  -p 24000:4000 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /Users/asher/develop/code/github/ACloudViewer:/opt/ACloudViewer/ACloudViewer \
  -v /Users/asher/develop/code/github/docker_cache/build:/opt/ACloudViewer/build \
  -v /Users/asher/develop/code/github/docker_cache/install:/opt/ACloudViewer/install \
  -v /Users/asher/develop/code/github/ACloudViewer/3rdparty_downloads:/opt/ACloudViewer/thirdparties \
  -v /Users/asher/develop/code/github/CloudViewer-ML:/opt/ACloudViewer/CloudViewer-ML \
  registry.cn-shanghai.aliyuncs.com/asher-ai/cloudviewer-deps:latest

# attach into container instance
docker exec -it cloudviewer_env /bin/bash

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
  -DBUILD_WEBRTC=ON
  -DBUILD_JUPYTER_EXTENSION=ON
  -DBUILD_LIBREALSENSE=ON
  -DBUILD_AZURE_KINECT=ON
  -DBUILD_RPC_INTERFACE=ON
  -DBUILD_BENCHMARKS=OFF
  -DWITH_OPENMP=ON
  -DUSE_SIMD=ON
  -DWITH_SIMD=ON
  -DBUILD_RECONSTRUCTION=ON
  -DGLIBCXX_USE_CXX11_ABI=0
  -DBUILD_CUDA_MODULE=ON
  -DBUILD_PYTORCH_OPS=OFF
  -DBUILD_TENSORFLOW_OPS=OFF
  -DBUNDLE_CLOUDVIEWER_ML=OFF
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

cmakeGuiOptions=(
                -DDEVELOPER_BUILD=OFF
                -DCMAKE_BUILD_TYPE=Release
                -DBUILD_JUPYTER_EXTENSION=ON
                -DBUILD_LIBREALSENSE=ON
                -DBUILD_AZURE_KINECT=ON
                -DBUILD_RPC_INTERFACE=ON
                -DBUILD_BENCHMARKS=OFF
                -DWITH_OPENMP=ON
                -DBUILD_WEBRTC=ON
                -DUSE_SIMD=ON
                -DWITH_SIMD=ON
                -DBUILD_COLMAP_GUI=ON
                -DBUILD_RECONSTRUCTION=ON
                -DBUILD_OPENCV=ON
                -DUSE_SYSTEM_OPENCV=ON
                -DBUILD_CUDA_MODULE=ON
                -DBUILD_PYTORCH_OPS=OFF
                -DBUILD_TENSORFLOW_OPS=OFF
                -DBUNDLE_CLOUDVIEWER_ML=OFF
                -DGLIBCXX_USE_CXX11_ABI=1
                -DCVCORELIB_USE_CGAL=ON
                -DCVCORELIB_SHARED=ON
                -DCVCORELIB_USE_QT_CONCURRENT=ON
                -DOPTION_USE_GDAL=ON
                -DOPTION_USE_DXF_LIB=ON
                -DOPTION_USE_RANSAC_LIB=ON
                -DOPTION_USE_SHAPE_LIB=ON
                -DPLUGIN_IO_QADDITIONAL=ON
                -DPLUGIN_IO_QCORE=ON
                -DPLUGIN_IO_QCSV_MATRIX=ON
                -DPLUGIN_IO_QE57=ON
                -DPLUGIN_IO_QMESH=ON
                -DPLUGIN_IO_QPDAL=OFF
                -DPLUGIN_IO_QLAS=ON
                -DPLUGIN_IO_QPHOTOSCAN=ON
                -DPLUGIN_IO_QDRACO=ON
                -DPLUGIN_IO_QRDB=ON
                -DPLUGIN_IO_QRDB_FETCH_DEPENDENCY=ON
                -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON
                -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON
                -DPLUGIN_STANDARD_QANIMATION=ON
                -DQANIMATION_WITH_FFMPEG_SUPPORT=ON
                -DPLUGIN_STANDARD_QCANUPO=ON
                -DPLUGIN_STANDARD_QCOMPASS=ON
                -DPLUGIN_STANDARD_QCSF=ON
                -DPLUGIN_STANDARD_QFACETS=ON
                -DPLUGIN_STANDARD_QCLOUDLAYERS=ON
                -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON
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

DBUS_SYSTEM_BUS_ADDRESS=unix:path=/host/run/dbus/system_bus_socket \
    USER=ubuntu \
    PASSWD=ubuntu \
    UID=1000 \
    GID=1000 \
    TZ=Asia/Shanghai \
    LANG=zh_CN.UTF-8 \
    LC_ALL=zh_CN.UTF-8 \
    LANGUAGE=zh_CN.UTF-8 \
    DEBIAN_FRONTEND=noninteractive

groupadd -f $USER \
    && useradd --create-home --no-log-init -g $USER $USER \
    && usermod -aG sudo $USER \
    && echo "$USER:$PASSWD" | chpasswd \
    && chsh -s /bin/bash $USER \
    && usermod  --uid $UID $USER \
    && groupmod --gid $GID $USER \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers

# Install some dependences and xfce4 desktop
apt-get update --fix-missing -y \
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

cp docker_files/google-chrome-stable_current_amd64.deb /opt
docker_files/nomachine.deb /opt
apt-get install -yf ./google-chrome-stable_current_amd64.deb \
    && rm ./google-chrome-stable_current_amd64.deb \
    && apt-get install -y pulseaudio kmod \
    && mkdir -p /var/run/dbus \
    && dpkg -i ./nomachine.deb \
    && sed -i "s|#EnableClipboard both|EnableClipboard both |g" /usr/NX/etc/server.cfg \
    && sed -i '/DefaultDesktopCommand/c\DefaultDesktopCommand "/usr/bin/startxfce4"' /usr/NX/etc/node.cfg