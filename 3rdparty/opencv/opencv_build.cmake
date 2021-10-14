include(ExternalProject)

set(ZLIB_CMAKE_FLAGS -DZLIB_ROOT=${ZLIB_INCLUDE_DIRS}/../)
set(TBB_CMAKE_FLAGS -DTBB_INCLUDE_DIRS:PATH=${STATIC_TBB_INCLUDE_DIR} -DTBB_LIBRARIES=${STATIC_TBB_LIBRARIES})
set(PNG_CMAKE_FLAGS -DPNG_LIBRARY=${LIBPNG_LIB_DIR}/lib${LIBPNG_LIBRARIES}.a -DPNG_PNG_INCLUDE_DIR=${LIBPNG_INCLUDE_DIRS})

ExternalProject_Add(
  ext_opencv_contrib
  PREFIX opencv_contrib
  URL https://github.com/opencv/opencv_contrib/archive/4.5.1.zip
  URL_MD5 ddb4f64d6cf31d589a8104655d39c99b
  BUILD_ALWAYS 0
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property(ext_opencv_contrib SOURCE_DIR)
set(opencv_contrib_SOURCE_DIR ${SOURCE_DIR})

ExternalProject_Add(
  ext_opencv
  PREFIX opencv
  URL https://github.com/opencv/opencv/archive/4.5.1.zip
  URL_MD5 cc13d83c3bf989b0487bb3798375ee08
  UPDATE_COMMAND ""
  BUILD_IN_SOURCE ON
  BUILD_ALWAYS 0
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
    -DOPENCV_EXTRA_MODULES_PATH=${opencv_contrib_SOURCE_DIR}/modules
    ${ZLIB_CMAKE_FLAGS} ${TBB_CMAKE_FLAGS}
    ${TIFF_CMAKE_FLAGS} ${PNG_CMAKE_FLAGS}
    -DBUILD_SHARED_LIBS=OFF
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DWITH_TBB=ON
    -DWITH_FFMPEG=${BUILD_FFMPEG}
    -DBUILD_opencv_python2=OFF
    -DBUILD_opencv_python3=OFF
    -DWITH_GTK_2_X=OFF
    -DWITH_V4L=OFF
    -DINSTALL_C_EXAMPLES=OFF
    -DINSTALL_PYTHON_EXAMPLES=OFF
    -DBUILD_EXAMPLES=OFF
    -DWITH_QT=OFF
    -DWITH_OPENGL=OFF
    -DWITH_VTK=OFF
    -DWITH_OPENEXR=OFF  # Build error on IlmBase includes without "OpenEXR/" prefix
    -DENABLE_PRECOMPILED_HEADERS=OFF
    -DWITH_CUDA=OFF
    -DWITH_OPENCL=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_LIST=core,improc,photo,objdetect,video,imgcodecs,videoio,features2d,xfeatures2d,version
    <SOURCE_DIR>
  BUILD_COMMAND $(MAKE)

  DEPENDS opencv_contrib
          3rdparty_tbb
          3rdparty_libpng
)
ExternalProject_Get_Property(ext_opencv INSTALL_DIR)
set(OPENCV_CMAKE_FLAGS -DOpenCV_DIR=${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/opencv4 -DOPENCV_DIR=${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/opencv4)
set(OPENCV_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(OPENCV_LIB_DIR ${INSTALL_DIR}/lib)
set(OPENCV_LIBRARIES opencv)
