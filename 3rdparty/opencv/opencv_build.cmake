include(ExternalProject)

set(OPENCV_MAJOR_VERSION "4.1")
set(OPENCV_MINOR_VERSION "1")
set(OPENCV_VERSION_FILE "${OPENCV_MAJOR_VERSION}.${OPENCV_MINOR_VERSION}.zip")
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
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        CMAKE_ARGS
        ${ExternalProject_CMAKE_ARGS_hidden}
        # -DBUILD_SHARED_LIBS=$<$<PLATFORM_ID:Linux>:ON:OFF>
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_SOURCE_DIR}
        -DBUILD_opencv_contrib=OFF
        -DOPENCV_ENABLE_NONFREE=OFF
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DOPENCV_FORCE_3RDPARTY_BUILD=ON
        -DWITH_TBB=OFF
        -DWITH_FFMPEG=OFF
        -DBUILD_JASPER=ON
        -DBUILD_JPEG=ON            #编译opencv 3rdparty自带的libjpeg
        -DBUILD_PNG=ON             #编译opencv 3rdparty自带的libpng
        -DBUILD_TIFF=ON            #编译opencv 3rdparty自带的libtiff
        -DBUILD_ZLIB=ON            #编译opencv 3rdparty自带的libzlib
        -DBUILD_WEBP=ON            #编译opencv 3rdparty自带的libwebp
        -DBUILD_OPENEXR=ON         #编译opencv 3rdparty自带的openexr
        # -DBUILD_PROTOBUF=OFF      #编译opencv 3rdparty自带的libprotobuf
        # -DWITH_OPENEXR=ON  # Build error on IlmBase includes without "OpenEXR/" prefix
        -DBUILD_opencv_world=ON
        -DBUILD_opencv_core=ON
        -DBUILD_opencv_highgui=ON
        -DBUILD_opencv_imgcodecs=ON
        -DBUILD_opencv_imgproc=ON
        -DBUILD_opencv_features2d=OFF
        -DBUILD_opencv_flann=OFF
        -DBUILD_opencv_xfeatures2d=OFF
        -DBUILD_opencv_photo=OFF
        -DBUILD_opencv_calib3d=OFF
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
        -DBUILD_opencv_video=OFF
        -DBUILD_opencv_videoio=OFF
        -DBUILD_opencv_legacy=OFF
        -DWITH_GSTREAMER=OFF
        -DWITH_GTK=OFF
        -DWITH_GTK_2_X=OFF
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
        DEPENDS ext_opencv_contrib ext_eigen
)
ExternalProject_Get_Property(ext_opencv INSTALL_DIR)
set(OPENCV_CMAKE_FLAGS -DOpenCV_DIR=${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR}/cmake/opencv4 -DOPENCV_DIR=${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR}/cmake/opencv4)
set(OpenCV_INCLUDE_DIRS ${INSTALL_DIR}/include/opencv4/)
set(OpenCV_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
set(THIRPARTY_LIB_PATH ${OpenCV_LIB_DIR}/opencv4/3rdparty)

if(WIN32)
    set(CV_LIB_SUFFIX $<$<CONFIG:Debug>:d>)
else()
    set(CV_LIB_SUFFIX "")
endif()

# fix undefined symbols issues
set(OpenCV_LIBS opencv_world${CV_LIB_SUFFIX}
                ade${CV_LIB_SUFFIX}
                ittnotify${CV_LIB_SUFFIX}
                libprotobuf${CV_LIB_SUFFIX}
                quirc${CV_LIB_SUFFIX}
                zlib${CV_LIB_SUFFIX}
                libjasper${CV_LIB_SUFFIX}
                libjpeg-turbo${CV_LIB_SUFFIX}
                libpng${CV_LIB_SUFFIX}
                libtiff${CV_LIB_SUFFIX}
                libwebp${CV_LIB_SUFFIX}
                IlmImf${CV_LIB_SUFFIX}
)

add_custom_command(
    TARGET ext_opencv
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} 
        -DSOURCE_DIR=${THIRPARTY_LIB_PATH}
        -DDESTINATION_DIR=${OpenCV_LIB_DIR}
        "-DLIB_FILTERS=${OpenCV_LIBS}"
        "-DLIBRARY_SUFFIX=${CMAKE_STATIC_LIBRARY_SUFFIX}"
        -P ${CloudViewer_3RDPARTY_DIR}/opencv/copy_files.cmake
    VERBATIM
)