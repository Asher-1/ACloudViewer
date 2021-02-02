#!/bin/bash

# install some dependence on host pc
sudo apt-get install x11-xserver-utils && xhost +

# create container instance
docker run -dit --runtime=nvidia --name=cloudViewer \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --privileged \
  --env HTTP_PROXY="http://127.0.0.1:1089" \
  --env HTTPS_PROXY="http://127.0.0.1:1089" \
  -e DISPLAY=unix$DISPLAY \
  --net=host \
  -e GDK_SCALE \
  -e GDK_DPI_SCALE \
  -m 8G \
  -p 10022:22 \
  -p 14000:4000 \
  -v /etc/localtime:/etc/localtime:ro \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /media/yons/data/develop/pcl_projects/ErowCloudViewer/docker_cache/build:/opt/ErowCloudViewer/build \
  -v /media/yons/data/develop/pcl_projects/ErowCloudViewer/docker_cache/install:/opt/ErowCloudViewer/install \
  -v /media/yons/data/develop/pcl_projects/ErowCloudViewer/thirdparties:/opt/ErowCloudViewer/thirdparties \
  cloudviewer-deps:develop-ubuntu18.04-cuda101

# attach into container instance
docker exec -it cloudViewer /bin/bash

ln -s /opt/Qt5.14.2/5.14.2/gcc_64/lib/libQt5X11Extras.so.5.14.2 /usr/lib/libQt5X11Extras.so.5
export PATH="/opt/Qt5.14.2/5.14.2/gcc_64/bin:$PATH"
export LD_LIBRARY_PATH="/opt/Qt5.14.2/5.14.2/gcc_64/lib:$LD_LIBRARY_PATH"
export QT_PLUGIN_PATH="/opt/Qt5.14.2/5.14.2/gcc_64/plugins:$QT_PLUGIN_PATH"
export QML2_IMPORT_PATH="/opt/Qt5.14.2/5.14.2/gcc_64/qml:$QML2_IMPORT_PATH"

# Build ErowCloudViewer
export DISPLAY=:0  #DISPLAY must be consistent with host
cd /opt/ErowCloudViewer
mkdir build
cd build

cmakWhlOptions=(-DDEVELOPER_BUILD=OFF
  -DCMAKE_BUILD_TYPE=Release
  -DBUILD_JUPYTER_EXTENSION=ON
  -DBUILD_LIBREALSENSE=ON
  -DBUILD_AZURE_KINECT=ON
  -DBUILD_RPC_INTERFACE=ON
  -DBUILD_BENCHMARKS=OFF
  -DWITH_OPENMP=ON
  -DBUILD_CUDA_MODULE=ON
  -DBUILD_PYTORCH_OPS=ON
  -DBUILD_TENSORFLOW_OPS=ON
  -DBUNDLE_CLOUDVIEWER_ML=ON
  -DCLOUDVIEWER_ML_ROOT=/opt/ErowCloudViewer/CloudViewer-ML
  -DTHIRD_PARTY_DOWNLOAD_DIR=/opt/ErowCloudViewer/thirdparties
  -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python3.6
  -DPYTHON_IN_PATH=/root/miniconda3/bin/python3.6
  -DPYTHON_LIBRARY=/root/miniconda3/lib/libpython3.6m.so
  -DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake
  -DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake
)
cmake "/opt/ErowCloudViewer/ErowCloudViewer" \
      "${cmakWhlOptions[@]}" \
      -DCMAKE_INSTALL_PREFIX=/opt/ErowCloudViewer/install

# compile ErowCloudViewer pip-package
make "-j$(nproc)" pip-package
make "-j$(nproc)" conda-package
make install "-j$(nproc)"

cmakGuiOptions=(-DDEVELOPER_BUILD=OFF
                -DCMAKE_BUILD_TYPE=Release
                -DBUILD_JUPYTER_EXTENSION=ON
                -DBUILD_LIBREALSENSE=ON
                -DBUILD_AZURE_KINECT=ON
                -DBUILD_RPC_INTERFACE=ON
                -DBUILD_BENCHMARKS=OFF
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
                -DPLUGIN_IO_QPDAL=ON
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
                -DTHIRD_PARTY_DOWNLOAD_DIR=/opt/ErowCloudViewer/thirdparties
                -DPYTHON_EXECUTABLE=/root/miniconda3/bin/python3.6
                -DPYTHON_IN_PATH=/root/miniconda3/bin/python3.6
                -DPYTHON_LIBRARY=/root/miniconda3/lib/libpython3.6m.so
                -DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake
                -DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake
        )

cmake "/opt/ErowCloudViewer/ErowCloudViewer" \
      "${cmakGuiOptions[@]}" \
      -DCMAKE_INSTALL_PREFIX=/opt/ErowCloudViewer/install

make "-j$(nproc)"
make install "-j$(nproc)"