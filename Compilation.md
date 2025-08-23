# Compilation of ACloudViewer 3.25+ (with CMake)

WINDOWS: [BUILD_SHELL](scripts/build_win.py)

```
python .\scripts\build_win.py
```

MACOS: [BUILD_SHELL](scripts/build_macos.sh)

```
(VPN required)[fix librealsense downloading issues]
export https_proxy=http://127.0.0.1:7890 http_proxy=http://127.0.0.1:7890 all_proxy=socks5://127.0.0.1:7890
Remove build_realsense when call build_mac_wheel at scripts/build_macos_whl.sh

./scripts/build_macos.sh 2>&1 | tee build.log
```

[**Fast Docker build on Linux**](./docker/README.md)

LINUX:(Docker) [BUILD_SHELL](docker/build-release.sh) and [BUILD_SHELL_CONDA](docker/build-release-conda.sh)

```
./docker/build-release.sh

or

./docker/build-release-conda.sh
```

Linux:(Manually)

```
(Linux whl)
sudo apt install libxxf86vm-dev

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH
cd ACloudViewer
mkdir build
cd build
cmake -DDEVELOPER_BUILD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBREALSENSE=ON \
      -DBUILD_AZURE_KINECT=ON \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_OPENCV=OFF \
      -DWITH_OPENMP=ON \
      -DWITH_IPP=ON \
      -DWITH_SIMD=ON \
      -DUSE_SIMD=ON \
      -DBUILD_WEBRTC=ON \
      -DUSE_PCL_BACKEND=OFF \
      -DBUILD_FILAMENT_FROM_SOURCE=OFF \
      -DBUILD_JUPYTER_EXTENSION=ON \
      -DBUILD_RECONSTRUCTION=ON \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_COMMON_CUDA_ARCHS=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_CLOUDVIEWER_ML=ON \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
      -DCMAKE_INSTALL_PREFIX=/home/asher/develop/code/github/CloudViewer/install \
      -DCLOUDVIEWER_ML_ROOT=/home/asher/develop/code/github/CloudViewer/CloudViewer-ML \
      ..

make "-j$(nproc)" python-package
make "-j$(nproc)" pip-package
make "-j$(nproc)" install-pip-package
python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"
```

```
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/cmake:$LD_LIBRARY_PATH"
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH
(Linux APP)
cd ACloudViewer
mkdir build_app
cd build_app
cmake   -DDEVELOPER_BUILD=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        -DBUILD_LIBREALSENSE=OFF \
        -DBUILD_AZURE_KINECT=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DWITH_OPENMP=ON \
        -DWITH_IPP=ON \
        -DWITH_SIMD=ON \
        -DUSE_SIMD=ON \
        -DPACKAGE=ON \
        -DUSE_PCL_BACKEND=ON \
        -DBUILD_WEBRTC=OFF \
        -DBUILD_OPENCV=ON \
        -DBUILD_RECONSTRUCTION=ON \
        -DBUILD_CUDA_MODULE=ON \
        -DBUILD_COMMON_CUDA_ARCHS=ON \
        -DBUILD_PYTORCH_OPS=OFF \
        -DBUILD_TENSORFLOW_OPS=OFF \
        -DBUNDLE_CLOUDVIEWER_ML=OFF \
        -DGLIBCXX_USE_CXX11_ABI=ON \
        -DCVCORELIB_USE_CGAL=ON \
        -DCVCORELIB_SHARED=ON \
        -DCVCORELIB_USE_QT_CONCURRENT=ON \
        -DOPTION_USE_GDAL=OFF \
        -DOPTION_USE_DXF_LIB=ON \
        -DOPTION_USE_RANSAC_LIB=ON \
        -DOPTION_USE_SHAPE_LIB=ON \
        -DPLUGIN_IO_QDRACO=ON \
        -DPLUGIN_IO_QLAS=ON \
        -DPLUGIN_IO_QADDITIONAL=ON \
        -DPLUGIN_IO_QCORE=ON \
        -DPLUGIN_IO_QCSV_MATRIX=ON \
        -DPLUGIN_IO_QE57=ON \
        -DPLUGIN_IO_QMESH=ON \
        -DPLUGIN_IO_QPDAL=OFF \
        -DPLUGIN_IO_QPHOTOSCAN=ON \
        -DPLUGIN_IO_QRDB=ON \
        -DPLUGIN_IO_QFBX=OFF \
        -DPLUGIN_IO_QSTEP=OFF \
        -DPLUGIN_STANDARD_QCORK=OFF \
        -DPLUGIN_STANDARD_QJSONRPC=ON \
        -DPLUGIN_STANDARD_QCLOUDLAYERS=ON \
        -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON \
        -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON \
        -DPLUGIN_STANDARD_QANIMATION=ON \
        -DQANIMATION_WITH_FFMPEG_SUPPORT=ON \
        -DPLUGIN_STANDARD_QCANUPO=ON \
        -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON \
        -DPLUGIN_STANDARD_QCOMPASS=ON \
        -DPLUGIN_STANDARD_QCSF=ON \
        -DPLUGIN_STANDARD_QFACETS=ON \
        -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON \
        -DPLUGIN_STANDARD_QM3C2=ON \
        -DPLUGIN_STANDARD_QMPLANE=ON \
        -DPLUGIN_STANDARD_QPCL=ON \
        -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
        -DPOISSON_RECON_WITH_OPEN_MP=ON \
        -DPLUGIN_STANDARD_QRANSAC_SD=ON \
        -DPLUGIN_STANDARD_QSRA=ON \
        -DPLUGIN_STANDARD_3DMASC=ON \
        -DPLUGIN_STANDARD_QTREEISO=ON \
        -DPLUGIN_STANDARD_QVOXFALL=ON \
        -DPLUGIN_PYTHON=ON \
        -DBUILD_PYTHON_MODULE=ON \
        -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
        -DCMAKE_INSTALL_PREFIX=/home/asher/develop/code/github/CloudViewer/install \
        ..

Build: 
        make "-j$(nproc)"
        make install "-j$(nproc)"
```

MacOS:(Manually)

```
(MacOS wheel)

Setup env:
cp .ci/conda_macos.yml /tmp/conda_macos.yml
sed -i "" "s/3.8/3.11/g" /tmp/conda_macos.yml
conda env create -f /tmp/conda_macos.yml
conda activate python3.11

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH

cd ACloudViewer
mkdir build
cd build

cmake -DDEVELOPER_BUILD=OFF \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBREALSENSE=ON \
      -DBUILD_AZURE_KINECT=OFF \
      -DBUILD_BENCHMARKS=OFF \
      -DBUILD_OPENCV=OFF \
      -DWITH_OPENMP=ON \
      -DWITH_IPP=OFF \
      -DWITH_SIMD=ON \
      -DUSE_SIMD=ON \
      -DUSE_PCL_BACKEND=OFF \
      -DBUILD_FILAMENT_FROM_SOURCE=OFF \
      -DBUILD_WEBRTC=OFF \
      -DBUILD_JUPYTER_EXTENSION=OFF \
      -DBUILD_RECONSTRUCTION=ON \
      -DBUILD_CUDA_MODULE=OFF \
      -DBUILD_COMMON_CUDA_ARCHS=ON \
      -DBUILD_PYTORCH_OPS=ON \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_CLOUDVIEWER_ML=ON \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DBUILD_WITH_CONDA=ON \
      -DCONDA_PREFIX=$CONDA_PREFIX \
      -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
      -DCMAKE_INSTALL_PREFIX=~/cloudViewer_install \
      -DCLOUDVIEWER_ML_ROOT=/Users/asher/develop/code/github/CloudViewer-ML \
      ..

make "-j$(nproc)" python-package
make "-j$(nproc)" pip-package
make "-j$(nproc)" install-pip-package
python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"
```

```
(MacOS APP) 

Setup env:
cp .ci/conda_macos_cloudViewer.yml /tmp/conda_macos_cloudViewer.yml
sed -i "" "s/3.8/3.11/g" /tmp/conda_macos_cloudViewer.yml
conda env create -f /tmp/conda_macos_cloudViewer.yml
conda activate cloudViewer

export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig:$PKG_CONFIG_PATH
export PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/pkgconfig:$CONDA_PREFIX/lib/cmake:$PATH

cd ACloudViewer
mkdir build_app
cd build_app
cmake   -DDEVELOPER_BUILD=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        -DBUILD_LIBREALSENSE=OFF \
        -DBUILD_AZURE_KINECT=OFF \
        -DBUILD_BENCHMARKS=OFF \
        -DWITH_OPENMP=ON \
        -DWITH_IPP=OFF \
        -DWITH_SIMD=ON \
        -DUSE_SIMD=ON \
        -DPACKAGE=ON \
        -DUSE_PCL_BACKEND=ON \
        -DBUILD_WEBRTC=OFF \
        -DBUILD_OPENCV=ON \
        -DBUILD_RECONSTRUCTION=ON \
        -DBUILD_CUDA_MODULE=OFF \
        -DBUILD_COMMON_CUDA_ARCHS=ON \
        -DBUILD_PYTORCH_OPS=OFF \
        -DBUILD_TENSORFLOW_OPS=OFF \
        -DBUNDLE_CLOUDVIEWER_ML=OFF \
        -DGLIBCXX_USE_CXX11_ABI=ON \
        -DCVCORELIB_USE_CGAL=ON \
        -DCVCORELIB_SHARED=ON \
        -DCVCORELIB_USE_QT_CONCURRENT=ON \
        -DOPTION_USE_GDAL=OFF \
        -DOPTION_USE_DXF_LIB=ON \
        -DOPTION_USE_RANSAC_LIB=ON \
        -DOPTION_USE_SHAPE_LIB=ON \
        -DPLUGIN_IO_QDRACO=ON \
        -DPLUGIN_IO_QLAS=ON \
        -DPLUGIN_IO_QADDITIONAL=ON \
        -DPLUGIN_IO_QCORE=ON \
        -DPLUGIN_IO_QCSV_MATRIX=ON \
        -DPLUGIN_IO_QE57=ON \
        -DPLUGIN_IO_QMESH=ON \
        -DPLUGIN_IO_QPDAL=OFF \
        -DPLUGIN_IO_QPHOTOSCAN=ON \
        -DPLUGIN_IO_QRDB=ON \
        -DPLUGIN_IO_QFBX=OFF \
        -DPLUGIN_IO_QSTEP=OFF \
        -DPLUGIN_STANDARD_QCORK=OFF \
        -DPLUGIN_STANDARD_QJSONRPC=ON \
        -DPLUGIN_STANDARD_QCLOUDLAYERS=ON \
        -DPLUGIN_STANDARD_MASONRY_QAUTO_SEG=ON \
        -DPLUGIN_STANDARD_MASONRY_QMANUAL_SEG=ON \
        -DPLUGIN_STANDARD_QANIMATION=ON \
        -DQANIMATION_WITH_FFMPEG_SUPPORT=ON \
        -DPLUGIN_STANDARD_QCANUPO=ON \
        -DPLUGIN_STANDARD_QCOLORIMETRIC_SEGMENTER=ON \
        -DPLUGIN_STANDARD_QCOMPASS=ON \
        -DPLUGIN_STANDARD_QCSF=ON \
        -DPLUGIN_STANDARD_QFACETS=ON \
        -DPLUGIN_STANDARD_QHOUGH_NORMALS=ON \
        -DPLUGIN_STANDARD_QM3C2=ON \
        -DPLUGIN_STANDARD_QMPLANE=ON \
        -DPLUGIN_STANDARD_QPCL=ON \
        -DPLUGIN_STANDARD_QPOISSON_RECON=ON \
        -DPOISSON_RECON_WITH_OPEN_MP=ON \
        -DPLUGIN_STANDARD_QRANSAC_SD=ON \
        -DPLUGIN_STANDARD_QSRA=ON \
        -DPLUGIN_STANDARD_3DMASC=ON \
        -DPLUGIN_STANDARD_QTREEISO=ON \
        -DPLUGIN_STANDARD_QVOXFALL=ON \
        -DPLUGIN_PYTHON=ON \
        -DBUILD_PYTHON_MODULE=ON \
        -DBUILD_WITH_CONDA=ON \
        -DCONDA_PREFIX=$CONDA_PREFIX \
        -DCMAKE_PREFIX_PATH=$CONDA_PREFIX/lib \
        -DCMAKE_INSTALL_PREFIX=~/cloudViewer_install \
        -DCLOUDVIEWER_ML_ROOT=/Users/asher/develop/code/github/CloudViewer-ML \
        ..

Build: 
        make -j24
        make install -j24
```

## Debug wheel (Linux)

```
1. gdb python

2. run -c "import cloudViewer"

3. bt
```

## Debug wheel (MacOS)

```
1. lldb python

2. run -c "import cloudViewer"

3. bt
```