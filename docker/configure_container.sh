#!/bin/bash

# install some dependence on host pc
sudo apt-get install x11-xserver-utils && xhost +

# create container instance
docker run -dit --runtime=nvidia --privileged --name=cloudViewer \
           -v /etc/localtime:/etc/localtime:ro \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -e DISPLAY=unix$DISPLAY \
           -e GDK_SCALE \
           -e GDK_DPI_SCALE \
           -m 8G \
           -p 2222:22 \
           erowcloudviewer-deps:develop-ubuntu18.04-cuda10.2

# attach into container instance
docker exec -it cloudViewer /bin/bash

# Build ErowCloudViewer
cd /opt/ErowCloudViewer
mkdir build
cd build
cmake "/opt/ErowCloudViewer/ErowCloudViewer" \
                            -DCMAKE_BUILD_TYPE=Release \
                            -DBUILD_JUPYTER_EXTENSION=ON \
                            -DBUILD_LIBREALSENSE=ON \
                            -DBUILD_AZURE_KINECT=ON \
                            -DBUILD_RPC_INTERFACE=ON \
                            -DBUILD_BENCHMARKS=OFF \
                            -DWITH_OPENMP=ON \
                            -DBUILD_CUDA_MODULE=ON \
                            -DBUILD_PYTORCH_OPS=ON \
                            -DBUILD_TENSORFLOW_OPS=ON \
                            -DBUNDLE_CLOUDVIEWER_ML=ON \
                            -DCLOUDVIEWER_ML_ROOT=/opt/ErowCloudViewer/CloudViewer-ML \
                            -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python3.6 \
                            -DPYTHON_IN_PATH=/root/miniconda3/bin/python3.6 \
                            -DPYTHON_LIBRARY=/root/miniconda3/lib/libpython3.6m.so \
                            -DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake \
                            -DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake \
                            -DCMAKE_INSTALL_PREFIX=/opt/ErowCloudViewer/install

# compile ErowCloudViewer pip-package
make "-j$(nproc)"
make "-j$(nproc)" pip-package
make "-j$(nproc)" conda-package
make install "-j$(nproc)"

export PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin:$PATH
export LD_LIBRARY_PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib:$LD_LIBRARY_PATH
export QT_PLUGIN_PATH=/opt/Qt5.14.2/5.14.2/gcc_64/plugins:$QT_PLUGIN_PATH
export QML2_IMPORT_PATH=/opt/Qt5.14.2/5.14.2/gcc_64/qml:$QML2_IMPORT_PATH
