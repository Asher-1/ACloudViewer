# Compilation of ACloudViewer 3.3+ (with CMake)

[**Fast Docker build**](./docker/README.md)

## Prerequisites

1.  Clone the main repository and its submodules from the main git(hub) server: <https://github.com/cloudcompare/trunk>

    `git clone --recursive https://github.com/cloudcompare/trunk.git`

2.  Install [CMake](http://www.cmake.org) (3.0 or newer)
3.  Install Qt (http://www.qt.io/ - for *Linux/Mac OS X*: qt-sdk)
      * ACloudViewer 2.7 requires **Qt version 5.5** or newer

4. Make sure you have a C++11 compliant compiler (gcc 4.7+ / clang / Visual 2013 and newer)

*To compile the project with older versions of Qt (from 4.8 to 5.4) or with a non C++11 compliant compiler, you'll have to stick with the https://github.com/cloudcompare/trunk/releases/tag/v2.6.3.1 version*

## Generating the project

1. Launch CMake GUI (`cmake-qt-gui` on Linux, the CMake application on Mac OS X)
  - *(for more convenience, you should check the "Grouped" check-box)*
  - set the `Where is the source code` field to your local repository (for instance `C:\ACloudViewer\trunk`)
  - set the `Where to build the binaries` field to ... almost anywhere you want **apart from the same folder as above or the *Program Files* folder (on Windows)**. (for instance: `C:\ACloudViewer\build`)
  - click the `Configure` button
  - select the generator for the project  
    The following generators have been tested:
      - Visual 2013, 2015, 2017 (64 bits)
      - gcc (Linux 64 bits)
      - Unix Makefiles (Mac OS X)
      - CodeBlocks - Unix Makefiles (Mac OS X)
  - wait for CMake configuration/tests to finish...
  - on the first run you may have to manually set the **QT5_ROOT_PATH** variable. Make it point to your installation of Qt (on Windows it's where the 'bin' folder lies - e.g. *Qt\5.6\msvc2013_64*)

2. Before clicking on the 'Generate' button, you may want to set some more options. If you expand the `OPTION` group, you'll be able to set some general options:
  - `OPTION_BUILD_CC_VIEWER`: whether or not to build the ccViewer side project (ON by default)
  - `OPTION_SUPPORT_MAC_PDMS_FORMAT`: to add support for PDMS .mac scripts (*CAD format*)
  - `OPTION_USE_DXFLIB`: to add support for DXF files in ACloudViewer/ccViewer with **dxflib** - see [below](#optional-setup-for-dxflib-support)
  - `OPTION_USE_FBX_SDK`: to add support for FBX files in ACloudViewer/ccViewer with the official **FBX SDK** - see [below](#optional-setup-for-fbx-sdk-support)
  - `OPTION_USE_GDAL`: to add support for a lot of raster files in ACloudViewer/ccViewer with **GDAL** library - see [below](#optional-setup-for-gdal-support)
  - `OPTION_USE_LIBE57`: to add support for E57 files in ACloudViewer/ccViewer with **libE57** - see [below](#optional-setup-for-libe57-support)
  - `OPTION_USE_SHAPE_LIB`: to add support for SHP files in ACloudViewer/ccViewer
  - `OPTION_PDAL_LAS`: to add support for LAS files in ACloudViewer/ccViewer with **PDAL** - see [below](#optional-setup-for-las-using-pdal)

  The following are Windows-only options:
  - `OPTION_MP_BUILD`: for Visual Studio only *(multi-process build --> much faster but uses a lot of CPU power)*
  - `OPTION_SUPPORT_3D_CONNEXION_DEVICES`: for 3D mouses handling
  - `OPTION_USE_OCULUS_SDK`: to add support for the Oculus Rift SDK in ACloudViewer/ccViewer (*work in progress*)
  - `OPTION_USE_VISUAL_LEAK_DETECTOR`: to use the Visual Leak Detector library for MSVC (http://vld.codeplex.com/)

3.  If you expand the `INSTALL` group, you'll be able to select the plugin(s) you want to compile.  By default, none are selected and **none are required** to work with ACloudViewer. See http://www.cloudcompare.org/doc/wiki/index.php?title=Plugins.
  - qAnimation *(relies on ffmpeg - https://www.ffmpeg.org/ - to generate video files)*
  - qCork (see [below](#optional-setup-for-cork--mpir-support-for-qcork))
  - qDummy *(does nothing, template for developers)*
  - qCSV_MATRIX_IO *(to load CSV matrix files)*
  - qFacets
  - qGMMReg *(relies on VXL)*
  - qHPR
  - qPCL (requires PCL - see [below](#optional-setup-for-pcl-required-by-qpcl))
  - qPCV
  - qPoissonRecon *(note: be sure to update the PoissonRecon submodule - see above)*
  - qRansacSD *(mainly tested on Windows but works with flaws on Linux)*
  - qSRA

4. _(Mac OS X only)_

  If you are compiling and running locally, add `-DCC_MAC_DEV_PATHS` to the `CMAKE_CXX_FLAGS` in the `CMAKE` group.  This will look for the plugins in your build directory rather than the application bundle.  If you need the shaders as well, you will have to create a `shaders` folder in the build directory and copy the shaders you need into it.

5.  Last but not least, the `CMAKE` group contains a `CMAKE_INSTALL_PREFIX` variable which is where ACloudViewer and ccViewer will be installed (when you compile the `INSTALL` project)
  - On Linux, default install dir is `/usr/local` (be sure to have administrative rights if you want to install ACloudViewer there: once configured, you can call `# make install` from the sources directory)
  - On Windows 7/8/10 CMake doesn't have the rights to 'install' files in the `Program Files` folder (even though it's CMake's default installation destination!)

## Generate the project files

Once all CMake errors have been resolved (you may to click multiple times on `Configure` if necessary)  be sure to click on the 'Generate' at least once at the end. This will create the project files for the compiler/IDE you have selected at the beginning. **At this point the project still needs to be compiled**.

## Compiling the project

Eventually you can run the compiler on the generated cmake file or open the project (e.g. for Visual Studio). The resulting files should be where you told CMake to *build the binaries* (e.g. `C:\ACloudViewer\build`).

*You should always find the two following configuration/sub-projects*:

1. `build all`: does all the compilation work (in the right order) but the binaries and libraries will be generated (by default) among all the other compilation files, in a somewhat complicated folder tree structure.
2.  `install`: copies all the necessary files (executable, resources, plugins, DLLs etc.) to the `CMAKE_INSTALL_PREFIX` folder. **This is mandatory to actually launch ACloudViewer or ccViewer.**


The Mac OS X install/release process is still a bit less automated than the others. If you are putting together a complete install (using `make install`), you will need to change the `PATH_PREFIX` variable in the script `libs/CVViewer/apps/fixup_macosx_bundle.sh`.  Please see the comment in that file and if you know how to solve it, please submit a patch.

### Working with Visual Studio on Windows

As all the files (executables, plugins and other DLLs) are copied in the `CMAKE_INSTALL_PREFIX` directory, the standard project launch/debug mechanism is broken. Therefore by default you won't be able to 'run' the ACloudViewer or ccViewer projects as is (with F5 or Ctrl + F5 for instance). See [this post](http://www.danielgm.net/cc/forum/viewtopic.php?t=992) on the forum to setup Visual correctly.

### Debugging plugins

If you want to use or debug plugins in DEBUG mode while using a single configuration compiler/IDE (gcc, etc.) the you'll have to comment the automatic definition of the `QT_NO_DEBUG` macro in '/plugins/CMakePluginTpl.cmake' (see http://www.cloudcompare.org/forum/viewtopic.php?t=2070).

### Install ACloudViewer Python package

Inside the activated virtualenv (shall be activated before ``cmake``),
run

    # 1) Create Python package
    # 2) Create pip wheel
    # 3) Install CloudViewer pip wheel the current virtualenv
    make install-pip-package

The above command is **compatible with both pip and Conda virtualenvs**. To
uninstall, run

    pip uninstall cloudViewer

If more fine-grained controls, here is a list of all related build targets:

    # Create Python package in build/lib/python_package
    make python-package

    # Create pip wheel in build/lib/python_package/pip_package
    make pip-package

    # Create conda package in build/lib/python_package/conda_package
    make conda-package

    # Install pip wheel
    make install-pip-package

If the installation is successful, we shall now be able to import cloudViewer

    python -c "import cloudViewer"

ACloudViewer can be installed as a C++ library or a Python package, by building the corresponding targets with Visual Studio or from the terminal. E.g.

`cmake --build . --parallel %NUMBER_OF_PROCESSORS% --config Release --target the-target-name` 

Here’s a list of installation related targets. Please refer to 5. Install for more detailed documentation.

1. install

2. python-package

3. pip-package

4. install-pip-package

### Translations

有道云翻译：
应用ID：6a470044a4c9069a
应用秘钥：62EddBoyfG0KXCp5Ih0C100Ir0DNYcId

翻译文件的正则替换：
ui_(\w+).h
\1.ui

"(\w+).cpp"
"../\1.cpp"
"(\w+).h"
"../\1.h"

Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Debug"
Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Debug\plugins"

Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Release"
Copy "$(TargetDir)*.*" "$(SolutionDir)\eCV\Release\plugins"

Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Debug"
Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Debug\plugins"
Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Release"
Copy "$(TargetDir)*.*" "$(SolutionDir)\qCC\Release\plugins"

# Appendix

## Additional optional CMake setup steps

### [Optional] Setup for the qPoissonRecon plugin

1. The version of the Poisson Surface Reconstruction library (M. Kazhdan et al.) used by the  is https://github.com/cloudcompare/PoissonRecon. It is declared as a submodule of CC's repository. You have to explicitly synchronize it (see https://git-scm.com/docs/git-submodule).
2. Then simply check the INSTALL_QPOISSON_RECON_PLUGIN option in CMake

### qLASFWIO [Deprecated] | qLASIO [recommended]

LAS/LAZ file support on Windows can also be achieved by compiling
[LASlib](https://github.com/CloudCompare/LAStools) and setting the following variables:

-   `LASLIB_INCLUDE_DIR`: LAStools/LASlib/inc
-   `LASZIP_INCLUDE_DIR`: LAStools/LASzip/src
-   `LASLIB_RELEASE_LIBRARY` or `LASLIB_DEBUG_LIBRARY`, depending on build type: the compiled library, e.g. `LAStools/LASlib/VC14/lib/x64/LASlibVC14.lib`

`Note`: 
```
brew install laszip [on macos]
https://github.com/LASzip/LASzip/releases/download/3.4.3/laszip-src-3.4.3.tar.gz [compile from source on linux]
```


### [Optional] Setup for LAS using PDAL [Deprecated]

If you want to compile ACloudViewer (and ccViewer) with LAS/LAZ files support, you'll need:

1. [PDAL](https://pdal.io/) ("sudo apt-get install libjsoncpp-dev -y" for LINUX) [sudo ln -s /usr/include/jsoncpp/json/ /usr/include/json]
		conda install -c conda-forge pdal python-pdal
2. Note: should fix libtiff.so load order bugs with opencv && pdal version should more than 2.0.0
3. Set `OPTION_PDAL_LAS=TRUE`

If your PDAL installation is not correctly picked up by CMake, 
set the `PDAL_DIR` to the path containing `PDALConfig.cmake`.
eg: "/home/yons/anaconda3/envs/pytorch-gpu/lib/cmake/PDAL"

### [Optional] Setup for LibE57 support

If you want to compile ACloudViewer (and ccViewer) with LibE57 files support, you'll need:

1. [Boost](http://www.boost.org/) multi-thread static libraries
2. [Xerces-C++](http://xerces.apache.org/xerces-c) multi-thread **static** libraries
    - On Visual C++ (Windows):
        1. select the `Static Debug` or `Static Release` configurations
        2. you'll have to manually modify the `XercesLib` project options so that the `C/C++ > Code Generation > Runtime Library` are of DLL type in both release and debug modes (i.e. `/MD` in release or `/MDd` in debug)
        3. for 64 bits version be sure to select the right platform (x64 instead of Win32). If you use Visual Studio Express 2010, be sure also that thif(NOT DEFINED Python3_FIND_REGISTRY)
#     # Only consider PATH variable on Windows by default
#     set(Python3_FIND_REGISTRY NEVER)
# endif()e `toolset` (in the project properties) is set to something like `Windows7.1SDK`
    - only the XercesLib project neet to be compiled
    - eventually, CMake will look for the resulting files in `/include` (instead of `/src`) and `/lib` (without the Release or Debug subfolders). By default the visual project will put them in `/Build/WinXX/VCXX/StaticXXX`. Therefore you should create a custom folder with the right organization and copy the files there.

    - On Linux
	wget https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.3.zip
	cd /opt \
	    && unzip xerces-c-3.2.3.zip \
	    && cd ./xerces-c-3.2.3 \
	    && chmod +x configure \
	    && ./configure --prefix=/usr \
	    && make \
	    && make install

3. [LibE57](http://libe57.org) (*last tested version: 1.1.312 on Windows*)
    - **WARNING**: with Visual Studio (at least), you'll need the libraries compiled with `/MD` (=DLL Multithreaded) in release mode and `/MDd` in debug mode. You may have to replace all `/MT` by `/MD` in the main libE57 root CMake file (or in `cmake/c_flag_overrides.cmake` and `cmake/cxx_flag_overrides.cmake` if there's no `/MT` in it)
    - If you found `set(Boost_USE_STATIC_RUNTIME ON)` in the CMake file, comment it
    - **the version 1.1.312 of libE57 has a small glitch that must be manually patched**:
        1.  open `E57FoundationImpl.cpp` and browse to the `CheckedFile::operator<<(float f)` method (line 4670)
        2.  set the output precision to 8 instead of 7! (otherwise the interal checks for precision loss may fail and libE57 will throw an exception)

The ACloudViewer CMake project will only require that you set the path where libE57 has been installed (`LIBE57_INSTALL_DIR`)

### [Optional] Setup for PCL (required by qPCL)

If you want to compile qPCL you'll need [PCL](http://pointclouds.org/) (*last tested version: 1.8 on Windows and 1.6 on Linux*)

Follow the online instructions/tutorials. Basically, you'll need Boost, Qt, Flann and Eigen.

Once properly installed, the ACloudViewer CMake script should automatically find PCL definitions. However, you'll have to set again the parameters related to Flann and Eigen.

### [Optional] Setup for FBX SDK support

If you want to compile ACloudViewer (and ccViewer) with FBX files support, you'll need: The official [Autodesk's FBX SDK](http://usa.autodesk.com/adsk/servlet/pc/item?siteID=123112&id=10775847) (last tested version: 2015.1 on Windows)

Then, the ACloudViewer CMake project will request that you set the 3 following variables:

1. `FBX_SDK_INCLUDE_DIR`: FBX SDK include directory (pretty straightforward ;)
2. `FBX_SDK_LIBRARY_FILE`: main FBX SDK library (e.g. `libfbxsdk-md.lib`)
3. `FBX_SDK_LIBRARY_FILE_DEBUG`: main FBX SDK library for debug mode (if any)

### [Optional] Setup for GDAL support

If you want to compile ACloudViewer (and ccViewer) with GDAL (raster) files support, you'll need a compiled version of the [GDAL library](http://www.gdal.org/) (last tested version: 1.10 on Windows, 2.0.2 on Mac OS X)
Then, the ACloudViewer CMake project will request that you set the 2 following variables:
0. conda install -c conda-forge gdal [for linux and windows]; brew install gdal [for macos]
1. `GDAL_INCLUDE_DIR`: GDAL include directory (pretty straightforward ;)
2. `GDAL_LIBRARY`: the static library (e.g. `gdal_i.lib`)

### [Optional] Setup for CGAL support
1. on Windows just set CGAL_DIR and set CGAL_Boost_USE_STATIC_LIBS = ON for fixing CGAL bugs.
2. on Linux use [sudo apt install libcgal-dev]

### [Optional] Setup for Cork + MPIR support (for qCork)

If you want to compile the qCork plugin (**on Windows only for now**), you'll need:

1. [MPIR 2.6.0](http://www.mpir.org/)
2. the forked version of the Cork library for CC: [<https://github.com/cloudcompare/cork>](https://github.com/cloudcompare/cork)
    - on Windows see the Visual project shipped with this fork and corresponding to your version (if any ;)
    - for VS2013 just edit the `mpir` property sheet (in the Properties manager) and update the MPIR macro (in the `User macros` tab)

Then, the ACloudViewer CMake project will request that you set the following variables:

1. `CORK_INCLUDE_DIR` and `MPIR_INCLUDE_DIR`: both libraries include directories (pretty straightforward ;)
2. `CORK_RELEASE_LIBRARY_FILE` and `MPIR_RELEASE_LIBRARY_FILE`: both main library files
3. and optionally `CORK_DEBUG_LIBRARY_FILE` and `MPIR_DEBUG_LIBRARY_FILE`: both main library files (for debug mode)


LINUX:

```
(whl)
sudo apt install libxxf86vm-dev

cd ACloudViewer
mkdir build
cd build
cmake -DDEVELOPER_BUILD=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_LIBREALSENSE=OFF \
      -DBUILD_AZURE_KINECT=ON \
      -DBUILD_BENCHMARKS=OFF \
      -DWITH_OPENMP=ON \
      -DWITH_IPPICV=ON \
      -DWITH_SIMD=ON \
      -DUSE_SIMD=ON \
      -DBUILD_WEBRTC=ON \
      -DBUILD_FILAMENT_FROM_SOURCE=OFF \
      -DBUILD_JUPYTER_EXTENSION=ON \
      -DBUILD_RECONSTRUCTION=ON \
      -DBUILD_CUDA_MODULE=ON \
      -DBUILD_COMMON_CUDA_ARCHS=ON \
      -DBUILD_PYTORCH_OPS=OFF \
      -DBUILD_TENSORFLOW_OPS=OFF \
      -DBUNDLE_CLOUDVIEWER_ML=OFF \
      -DGLIBCXX_USE_CXX11_ABI=OFF \
      -DCMAKE_INSTALL_PREFIX=/home/asher/develop/code/github/CloudViewer/install \
      -DCLOUDVIEWER_ML_ROOT=/home/asher/develop/code/github/CloudViewer/CloudViewer-ML \
      -DQT_QMAKE_EXECUTABLE:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/bin/qmake \
      -DCMAKE_PREFIX_PATH:PATH=/opt/Qt5.14.2/5.14.2/gcc_64/lib/cmake \
      ..

make "-j$(nproc)" python-package
make "-j$(nproc)" pip-package
make "-j$(nproc)" install-pip-package
python3 -c "import cloudViewer as cv3d; print(cv3d.__version__)"
```

```
(APP)
cd ACloudViewer
mkdir build
cd build
cmake   -DDEVELOPER_BUILD=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_JUPYTER_EXTENSION=OFF \
        -DBUILD_LIBREALSENSE=OFF \
        -DBUILD_AZURE_KINECT=ON \
        -DBUILD_BENCHMARKS=OFF \
        -DWITH_OPENMP=ON \
        -DWITH_IPPICV=ON \
        -DWITH_SIMD=ON \
        -DUSE_SIMD=ON \
        -DPACKAGE=ON \
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
        -DOPTION_USE_GDAL=ON \
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
        -DCMAKE_INSTALL_PREFIX=/home/asher/develop/code/github/CloudViewer/install \
        ..

Build: 
        make -j24
        make install -j24
```
