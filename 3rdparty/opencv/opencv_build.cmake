include(ExternalProject)

set(OPENCV_VERSION_FILE "4.1.1.zip")
file(GLOB PATCH_FILES "${CloudViewer_3RDPARTY_DIR}/opencv/boostdesc_bgm/*.i")
ExternalProject_Add(
        ext_opencv_contrib
        PREFIX opencv_contrib
        URL https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION_FILE}
        URL_MD5 a00c9f7edde2631759f6ed18e17d8dfb
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/opencv_contrib"
        BUILD_ALWAYS 0
        UPDATE_COMMAND ""
        CONFIGURE_COMMAND ""
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${PATCH_FILES} <SOURCE_DIR>/modules/xfeatures2d/src
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_opencv_contrib SOURCE_DIR)
set(OPENCV_CONTRIB_SOURCE_DIR "${SOURCE_DIR}/modules")

ExternalProject_Add(
        ext_opencv
        PREFIX opencv
        URL https://github.com/opencv/opencv/archive/${OPENCV_VERSION_FILE}
        URL_MD5 aa6df0e554f27d5a707ead76f050712b
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/opencv"
        UPDATE_COMMAND ""
        BUILD_IN_SOURCE OFF
        BUILD_ALWAYS 0
        CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DBUILD_SHARED_LIBS=ON
        -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_SOURCE_DIR}
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        -DOPENCV_ENABLE_NONFREE=ON
        -DWITH_TBB=OFF
        -DWITH_FFMPEG=OFF
        -DBUILD_JASPER=ON
        -DBUILD_JPEG=ON            #编译opencv 3rdparty自带的libjpeg
        -DBUILD_PNG=ON             #编译opencv 3rdparty自带的libpng
        -DBUILD_PROTOBUF=OFF       #编译opencv 3rdparty自带的libprotobuf
        -DBUILD_TIFF=ON            #编译opencv 3rdparty自带的libtiff
        -DBUILD_ZLIB=ON            #编译opencv 3rdparty自带的libzlib
        -DBUILD_WEBP=ON            #编译opencv 3rdparty自带的libwebp
        -DBUILD_opencv_world=ON
        -DBUILD_opencv_core=ON
        -DBUILD_opencv_highgui=ON
        -DBUILD_opencv_imgcodecs=ON
        -DBUILD_opencv_imgproc=ON
        -DBUILD_opencv_calib3d=ON
        -DBUILD_opencv_features2d=ON
        -DBUILD_opencv_flann=ON
        -DBUILD_opencv_photo=ON
        -DWITH_OPENEXR=ON  # Build error on IlmBase includes without "OpenEXR/" prefix
        -DBUILD_opencv_xfeatures2d=ON
        -DBUILD_JAVA=OFF
        -DBUILD_opencv_sfm=OFF # disabled ceres dependence compiling issues [only support 1.x.x for ceres]
        -DBUILD_opencv_apps=OFF
        -DBUILD_opencv_python2=OFF
        -DBUILD_opencv_python3=OFF
        -DBUILD_opencv_python_bindings_generator=OFF
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_opencv_gapi=OFF
        -DBUILD_opencv_java_bindings_generator=OFF
        -DBUILD_opencv_face=OFF
        -DBUILD_opencv_js=OFF
        -DBUILD_opencv_dnn=OFF
        -DBUILD_opencv_ml=OFF
        -DBUILD_opencv_objdetect=OFF
        -DBUILD_opencv_xobjdetect=OFF
        -DBUILD_opencv_dnn_objdetect=OFF
        -DBUILD_opencv_optflow=OFF
        -DBUILD_opencv_stitching=OFF
        -DBUILD_opencv_ts=OFF
        -DBUILD_opencv_video=ON
        -DBUILD_opencv_videoio=ON
        -DWITH_GTK=ON
        -DWITH_GTK_2_X=ON
        -DWITH_V4L=OFF
        -DWITH_CAROTENE=OFF
        -DWITH_OPENGL=OFF
        -DWITH_OPENCL=OFF
        -DWITH_LAPACK=OFF
        -DENABLE_PRECOMPILED_HEADERS=OFF
        -DINSTALL_C_EXAMPLES=OFF
        -DINSTALL_PYTHON_EXAMPLES=OFF
        -DBUILD_EXAMPLES=OFF
        -DWITH_QT=OFF
        -DWITH_IPP=OFF # disabled ippicv acceleration on intel chips
        -DWITH_VTK=OFF
        -DWITH_CUDA=OFF
        -DBUILD_TESTS=OFF
        # -DBUILD_LIST=core,improc,photo,objdetect,video,imgcodecs,videoio,features2d,version
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        DEPENDS ext_opencv_contrib
)
ExternalProject_Get_Property(ext_opencv INSTALL_DIR)
set(OPENCV_CMAKE_FLAGS -DOpenCV_DIR=${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR}/cmake/opencv4 -DOPENCV_DIR=${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR}/cmake/opencv4)
set(OPENCV_INCLUDE_DIRS ${INSTALL_DIR}/include/opencv4/)
set(OPENCV_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})

if (WIN32)
    set(EXT_CERES_LIBRARIES opencv_world$<$<CONFIG:Debug>:d>)
else ()
    set(OPENCV_LIBRARIES opencv_world)
endif ()
