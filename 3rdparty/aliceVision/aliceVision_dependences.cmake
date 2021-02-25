# ==============================================================================
# Use CMake ExternalProject to build all dependencies
# ==============================================================================
include(ExternalProject)

option(AV_BUILD_TIFF "Enable building an embedded Tiff" 		ON)
option(AV_BUILD_JPEG "Enable building an embedded Jpeg"                 ON)
option(AV_BUILD_LIBRAW "Enable building an embedded libraw" 		ON)
option(AV_BUILD_POPSIFT "Enable building an embedded PopSift" 		ON)
option(AV_BUILD_CCTAG "Enable building an embedded CCTag" 		ON)
option(AV_BUILD_OPENGV "Enable building an embedded OpenGV" 		ON)
option(AV_BUILD_OPENCV "Enable building an embedded OpenCV" 		ON)
option(AV_BUILD_LAPACK "Enable building an embedded Lapack" 		ON)
option(AV_BUILD_SUITESPARSE "Enable building an embedded SuiteSparse" 	ON)
option(AV_BUILD_FFMPEG "Enable building an embedded FFMpeg" 		ON)

if(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Force CMAKE_INSTALL_DO_STRIP in Release")
    set(CMAKE_INSTALL_DO_STRIP TRUE)
endif()
set(BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/external")

set(CMAKE_CORE_BUILD_FLAGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS} -DCMAKE_INSTALL_DO_STRIP:BOOL=${CMAKE_INSTALL_DO_STRIP} -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER})

set(ZLIB_CMAKE_FLAGS -DZLIB_ROOT=${ZLIB_INCLUDE_DIRS}/../)
set(EIGEN_CMAKE_FLAGS ${EIGEN_DISABLE_ALIGN_FLAGS} -DEigen3_DIR:PATH=${EIGEN_ROOT_DIR}/share/eigen3/cmake -DEIGEN3_INCLUDE_DIR=${EIGEN_INCLUDE_DIR} -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR} -DEigen_INCLUDE_DIR=${EIGEN_INCLUDE_DIR})
set(TBB_CMAKE_FLAGS -DTBB_INCLUDE_DIRS:PATH=${STATIC_TBB_INCLUDE_DIR} -DTBB_LIBRARIES=${STATIC_TBB_LIBRARIES})
set(PNG_CMAKE_FLAGS -DPNG_LIBRARY=${LIBPNG_LIB_DIR}/lib${LIBPNG_LIBRARIES}.a -DPNG_PNG_INCLUDE_DIR=${LIBPNG_INCLUDE_DIRS})
set(BOOST_CMAKE_FLAGS -DBOOST_ROOT=${BOOST_INCLUDE_DIRS})
if(CUDA_TOOLKIT_ROOT_DIR)
  message("CUDA_TOOLKIT_ROOT_DIR: " ${CUDA_TOOLKIT_ROOT_DIR})
  set(CUDA_CMAKE_FLAGS -DCUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR})
endif()

# Add Geogram
if(WIN32)
   set(VORPALINE_PLATFORM Win-vs-dynamic-generic)
elseif(APPLE)
   set(VORPALINE_PLATFORM Darwin-clang-dynamic)
elseif(UNIX)
   set(VORPALINE_PLATFORM Linux64-gcc-dynamic)
endif()

set(GEOGRAM_TARGET geogram)
ExternalProject_Add(${GEOGRAM_TARGET}
       URL https://github.com/alicevision/geogram/archive/v1.7.6.tar.gz
       URL_HASH MD5=d0f138f1cd50c633c30eacdb03619211
       DOWNLOAD_DIR ${BUILD_DIR}/download/geogram
       PREFIX ${GEOGRAM_TARGET}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND ${CMAKE_COMMAND} ${CMAKE_CORE_BUILD_FLAGS}
                -DVORPALINE_PLATFORM=${VORPALINE_PLATFORM}
                -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                -DBUILD_SHARED_LIBS=ON
                -DGEOGRAM_WITH_HLBFGS=OFF
                -DGEOGRAM_WITH_TETGEN=OFF
                -DGEOGRAM_WITH_GRAPHICS=OFF
                -DGEOGRAM_WITH_EXPLORAGRAM=OFF
                -DGEOGRAM_WITH_LUA=OFF
                -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> <SOURCE_DIR>
       BUILD_COMMAND $(MAKE)
       DEPENDS ${ZLIB_TARGET}
       )
ExternalProject_Get_Property(${GEOGRAM_TARGET} INSTALL_DIR)
set(GEOGRAM_CMAKE_FLAGS -DGEOGRAM_INSTALL_PREFIX=${INSTALL_DIR} -DGEOGRAM_INCLUDE_DIR=${INSTALL_DIR}/include/geogram1)
set(GEOGRAM_INCLUDE_DIRS ${INSTALL_DIR}/include/geogram1/)
set(GEOGRAM_LIB_DIR ${INSTALL_DIR}/lib)
set(GEOGRAM_LIBRARIES ${GEOGRAM_TARGET} ${GEOGRAM_TARGET}_num_3rdparty)

if(AV_BUILD_OPENGV)
set(OPENGV_TARGET opengv)
ExternalProject_Add(${OPENGV_TARGET}
       GIT_REPOSITORY https://github.com/laurentkneip/opengv.git
       GIT_TAG 91f4b19c73450833a40e463ad3648aae80b3a7f3
       PREFIX ${OPENGV_TARGET}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND ${CMAKE_COMMAND}
                  ${CMAKE_CORE_BUILD_FLAGS}
                  ${EIGEN_CMAKE_FLAGS}
                  -DCMAKE_POSITION_INDEPENDENT_CODE=ON
                  -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
       BUILD_COMMAND $(MAKE)
       DEPENDS ${EIGEN3_TARGET}
       )

ExternalProject_Get_Property(${OPENGV_TARGET} INSTALL_DIR)
set(OPENGV_CMAKE_FLAGS -DOPENGV_DIR=${INSTALL_DIR})
set(OPENGV_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(OPENGV_LIB_DIR ${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})
set(OPENGV_LIBRARIES ${OPENGV_TARGET})
endif()

if(AV_BUILD_LAPACK)
set(LAPACK_TARGET lapack)
ExternalProject_Add(${LAPACK_TARGET}
       #   http://www.netlib.org/lapack/lapack-3.9.0.tar.gz
       URL https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz
       URL_HASH MD5=0b251e2a8d5f949f99b50dd5e2200ee2
       DOWNLOAD_DIR ${BUILD_DIR}/download/lapack
       PREFIX ${LAPACK_TARGET}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND ${CMAKE_COMMAND}
            ${CMAKE_CORE_BUILD_FLAGS}
            -DBUILD_SHARED_LIBS=ON
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
       BUILD_COMMAND $(MAKE)
       DEPENDS ${TBB_TARGET}
       )

ExternalProject_Get_Property(${LAPACK_TARGET} INSTALL_DIR)
set(BLAS_LIBRARIES ${INSTALL_DIR}/lib/libblas.so)
set(LAPACK_LIBRARIES ${INSTALL_DIR}/lib/liblapack.so)
set(LAPACK_CMAKE_FLAGS -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES})

set(LAPACK_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(LAPACK_LIB_DIR ${INSTALL_DIR}/lib)
set(LAPACKBLAS_LIBRARIES ${LAPACK_TARGET} blas)
endif()

if(AV_BUILD_SUITESPARSE)

ExternalProject_add(gmp
      URL https://gmplib.org/download/gmp/gmp-6.2.1.tar.xz
      URL_HASH MD5=0b82665c4a92fd2ade7440c13fcaa42b
      DOWNLOAD_DIR ${BUILD_DIR}/download/gmp
      PREFIX gmp
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --enable-cxx
      BUILD_COMMAND $(MAKE)
)
ExternalProject_add(mpfr
      URL https://ftp.gnu.org/gnu/mpfr/mpfr-4.1.0.tar.gz
      URL_HASH MD5=81a97a9ba03590f83a30d26d4400ce39
      DOWNLOAD_DIR ${BUILD_DIR}/download/mpfr
      PREFIX mpfr
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --with-gmp=<INSTALL_DIR>
      BUILD_COMMAND $(MAKE)
      DEPENDS gmp
)

ExternalProject_Get_Property(gmp INSTALL_DIR)
set(GMP_INSTALL_DIR ${INSTALL_DIR})
ExternalProject_Get_Property(mpfr INSTALL_DIR)
set(MPFR_INSTALL_DIR ${INSTALL_DIR})

set(SUITESPARSE_TARGET suitesparse)
set(SUITESPARSE_INTERNAL_MAKE_CMD MPFR_ROOT=${MPFR_INSTALL_DIR} GMP_ROOT=${GMP_INSTALL_DIR} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${GMP_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}:${MPFR_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR} $(MAKE) BLAS="${BLAS_LIBRARIES}" LAPACK="${LAPACK_LIBRARIES}")
ExternalProject_Add(${SUITESPARSE_TARGET}
       # URL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v4.5.6.tar.gz
       # URL https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/v5.8.1.tar.gz  # requires gxx >= 4.9, centos 7 use gxx-4.8.5 by default
       # URL_HASH MD5=c414679bbc9432a3def01b31ad921140
       GIT_REPOSITORY https://github.com/alicevision/SuiteSparse
       GIT_TAG fix/gmp_mpfr  # based on v5.8.1
       # DOWNLOAD_DIR ${BUILD_DIR}/download/suitesparse
       PREFIX ${SUITESPARSE_TARGET}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND ""
       PATCH_COMMAND sh -c "test `nvcc --version | grep -o 'release [0-9]*' | grep -o '[0-9]*'` -gt 10" && sed -i /compute_30/d <SOURCE_DIR>/SuiteSparse_config/SuiteSparse_config.mk || true
       BUILD_COMMAND   cd <SOURCE_DIR> && ${SUITESPARSE_INTERNAL_MAKE_CMD} library CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
       INSTALL_COMMAND cd <SOURCE_DIR> && ${SUITESPARSE_INTERNAL_MAKE_CMD} install library INSTALL=<INSTALL_DIR> CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER}
       DEPENDS ${LAPACK_TARGET} mpfr
       )

ExternalProject_Get_Property(${SUITESPARSE_TARGET} INSTALL_DIR)
set(SUITESPARSE_CMAKE_FLAGS ${LAPACK_CMAKE_FLAGS} -DSUITESPARSE_INCLUDE_DIR_HINTS=${INSTALL_DIR}/include -DSUITESPARSE_LIBRARY_DIR_HINTS=${INSTALL_DIR}/lib)
set(SUITESPARSE_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(SUITESPARSE_LIB_DIR ${INSTALL_DIR}/lib)
set(SUITESPARSE_LIBRARIES   SuiteSparse_config
                            SuiteSparse_GPURuntime
                            GPUQREngine
                            mongoose amd btf
                            camd ccolamd colamd
                            cholmod cxsparse ldl
                            klu umfpack rbio spqr
                            graphblas sliplu)
endif()

# Add ceres-solver: A Nonlinear Least Squares Minimizer
set(CERES_TARGET ceres)
ExternalProject_Add(${CERES_TARGET}
    # TODO: update ceres to 2.0
    #URL https://github.com/ceres-solver/ceres-solver/archive/2.0.0.tar.gz
    #URL_HASH MD5=94246057ac520313e3b582c45a30db6e
       # URL https://github.com/ceres-solver/ceres-solver/archive/1.14.0.tar.gz
       GIT_REPOSITORY https://github.com/alicevision/ceres-solver
       GIT_TAG compatibility_gcc_4  # specific commit from the WIP 2.0 version with a fix for gcc-4
       PREFIX ${CERES_TARGET}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND ${CMAKE_COMMAND}
           ${CMAKE_CORE_BUILD_FLAGS}
           ${EIGEN_CMAKE_FLAGS}
           ${SUITESPARSE_CMAKE_FLAGS}
           -DSUITESPARSE:BOOL=ON
           -DLAPACK:BOOL=ON
           -DMINIGLOG=ON
           -DBUILD_TESTING:BOOL=OFF
           -DBUILD_EXAMPLES:BOOL=OFF
           -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
       BUILD_COMMAND $(MAKE)
       DEPENDS ${EIGEN3_TARGET} ${SUITESPARSE_TARGET}
       )

ExternalProject_Get_Property(${CERES_TARGET} INSTALL_DIR)
set(CERES_CMAKE_FLAGS ${SUITESPARSE_CMAKE_FLAGS} -DCeres_DIR=${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/Ceres)
set(CERES_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(CERES_LIB_DIR ${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR})
set(CERES_LIBRARIES ceres$<$<CONFIG:Debug>:-debug>)

## Add OpenEXR
set(OPENEXR_TARGET openexr)
ExternalProject_Add(${OPENEXR_TARGET}
      # vfxplatform CY2020: 2.4.x, but we use 2.5.x to avoid cmake issues
      #URL https://github.com/openexr/openexr/archive/v2.4.1.tar.gz
      #URL_HASH MD5=f7f7f893cf38786f88c306dec127113f
      URL https://github.com/AcademySoftwareFoundation/openexr/archive/v2.5.4.tar.gz
      URL_HASH MD5=e84577f884f05f7432b235432dfec455
      DOWNLOAD_DIR ${BUILD_DIR}/download/openexr
      # URL https://github.com/openexr/openexr/archive/v2.2.1.tar.gz
      # The release 2.2.1 has troubles with C++17, which breaks compilation with recent compilers.
      # The problem has been fixed https://github.com/openexr/openexr/issues/235
      # but there is no release yet, so we use the development version.
      # GIT_REPOSITORY https://github.com/openexr/openexr
      # Use the latest commit with g++4.X compatibility
      # GIT_TAG 74b5c1dc2dfbdce74987a57f5e011dc711f9ca65
      # Finally use a custom version for a cmake fix
      # GIT_REPOSITORY https://github.com/alicevision/openexr
      # GIT_TAG develop_compatibility_gxx4
      # GIT_REPOSITORY https://github.com/openexr/openexr
      # GIT_TAG a12937f6d7650d4fb81b469900ee2fd4c082c208
      PREFIX ${OPENEXR_TARGET}
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ${CMAKE_COMMAND}
                  ${CMAKE_CORE_BUILD_FLAGS}
                  ${ZLIB_CMAKE_FLAGS}
                  -DOPENEXR_BUILD_PYTHON_LIBS=OFF
                  -DOPENEXR_ENABLE_TESTS=OFF
                  -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> <SOURCE_DIR>
      BUILD_COMMAND $(MAKE)
      DEPENDS ${ZLIB_TARGET}
      )

ExternalProject_Get_Property(${OPENEXR_TARGET} INSTALL_DIR)
set(ILMBASE_CMAKE_FLAGS -DILMBASE_ROOT=${INSTALL_DIR} -DILMBASE_INCLUDE_PATH=${INSTALL_DIR}/include/OpenEXR)
set(OPENEXR_CMAKE_FLAGS ${ILMBASE_CMAKE_FLAGS} -DOPENEXR_ROOT=${INSTALL_DIR} -DOPENEXR_INCLUDE_PATH=${INSTALL_DIR}/include)
set(OPENEXR_INCLUDE_DIRS ${INSTALL_DIR}/include/ ${INSTALL_DIR}/include/OpenEXR)
set(OPENEXR_LIB_DIR ${INSTALL_DIR}/lib)
set(OPENEXR_LIBRARIES ${OPENEXR_TARGET})

# Add LibTiff
if(AV_BUILD_TIFF)
set(TIFF_TARGET tiff)
ExternalProject_Add(${TIFF_TARGET}
       URL http://download.osgeo.org/libtiff/tiff-4.2.0.tar.gz
       URL_HASH MD5=2bbf6db1ddc4a59c89d6986b368fc063
       DOWNLOAD_DIR ${BUILD_DIR}/download/tiff
       PREFIX ${TIFF_TARGET}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR>
       BUILD_COMMAND $(MAKE)
       DEPENDS ${ZLIB_TARGET}
       )

ExternalProject_Get_Property(${TIFF_TARGET} INSTALL_DIR)
SET(TIFF_CMAKE_FLAGS -DTIFF_LIBRARY=${INSTALL_DIR}/lib/libtiff.so -DTIFF_INCLUDE_DIR=${INSTALL_DIR}/include)
set(TIFF_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(TIFF_LIB_DIR ${INSTALL_DIR}/lib)
set(TIFF_LIBRARIES ${TIFF_TARGET} tiffxx)

endif()

if(AV_BUILD_JPEG)
set(JPEG_TARGET turbojpeg)
# Add turbojpeg
ExternalProject_Add(${JPEG_TARGET}
       URL https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.6.tar.gz
       URL_HASH MD5=22aad1e0772cd797306a87428dd744c7
       DOWNLOAD_DIR ${BUILD_DIR}/download/libjpeg-turbo
       PREFIX ${JPEG_TARGET}
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       # CONFIGURE_COMMAND cd <BINARY_DIR> && autoreconf -fiv && ./configure --prefix=<INSTALL_DIR>
       CONFIGURE_COMMAND ${CMAKE_COMMAND}
                       ${CMAKE_CORE_BUILD_FLAGS}
                       ${ZLIB_CMAKE_FLAGS}
                       -DBUILD_SHARED_LIBS=ON
                       -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> <SOURCE_DIR>
       BUILD_COMMAND $(MAKE)
       DEPENDS ${ZLIB_TARGET}
       )

   ExternalProject_Get_Property(${JPEG_TARGET} INSTALL_DIR)
   SET(JPEG_CMAKE_FLAGS -DJPEG_LIBRARY=${INSTALL_DIR}/lib/libjpeg.so -DJPEG_INCLUDE_DIR=${INSTALL_DIR}/include)
endif()

if(AV_BUILD_LIBRAW)
set(LIBRAW_TARGET libraw)
# Add libraw
ExternalProject_Add(libraw_cmake
      GIT_REPOSITORY https://github.com/LibRaw/LibRaw-cmake
      GIT_TAG master
      PREFIX libraw_cmake
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      )

ExternalProject_Get_Property(libraw_cmake SOURCE_DIR)
set(libraw_cmake_source_dir ${SOURCE_DIR})

ExternalProject_Add(${LIBRAW_TARGET}
      URL https://github.com/LibRaw/LibRaw/archive/0.20.0.tar.gz
      PREFIX ${LIBRAW_TARGET}
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      # Native libraw configure script doesn't work on centos 7 (autoconf 2.69)
      # CONFIGURE_COMMAND autoconf && ./configure --enable-jpeg --enable-openmp --disable-examples --prefix=<INSTALL_DIR>
      # Use cmake build system (not maintained by libraw devs)
      CONFIGURE_COMMAND cp ${libraw_cmake_source_dir}/CMakeLists.txt <SOURCE_DIR> &&
                cp -rf ${libraw_cmake_source_dir}/cmake <SOURCE_DIR> &&
                ${CMAKE_COMMAND}
                ${CMAKE_CORE_BUILD_FLAGS}
                ${ZLIB_CMAKE_FLAGS}
                -DBUILD_SHARED_LIBS=ON
                -DENABLE_OPENMP=ON
                -DENABLE_LCMS=ON
                -DENABLE_EXAMPLES=OFF
                -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
                -DINSTALL_CMAKE_MODULE_PATH:PATH=<INSTALL_DIR>/cmake <SOURCE_DIR>
      BUILD_COMMAND $(MAKE)
      DEPENDS libraw_cmake ${ZLIB_TARGET}
      )

ExternalProject_Get_Property(${LIBRAW_TARGET} INSTALL_DIR)
SET(LIBRAW_CMAKE_FLAGS  -DLIBRAW_PATH=${INSTALL_DIR}
                        -DPC_LIBRAW_INCLUDEDIR=${INSTALL_DIR}/include
                        -DPC_LIBRAW_LIBDIR=${INSTALL_DIR}/lib
                        -DPC_LIBRAW_R_LIBDIR=${INSTALL_DIR}/lib)

set(LIBRAW_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(LIBRAW_LIB_DIR ${INSTALL_DIR}/lib)
set(LIBRAW_LIBRARIES raw)

endif()

if(AV_BUILD_FFMPEG)

# GPL
# ExternalProject_add(x264
#       GIT_REPOSITORY https://code.videolan.org/videolan/x264.git
#       GIT_TAG 5493be84
#       GIT_PROGRESS ON
#       PREFIX ${BUILD_DIR}
#       BUILD_IN_SOURCE 0
#       BUILD_ALWAYS 0
#       UPDATE_COMMAND ""
#       INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
#       CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --disable-static --enable-shared
#       BUILD_COMMAND $(MAKE)
# )
# GPL
# ExternalProject_add(x265
#       GIT_REPOSITORY https://code.videolan.org/videolan/x265.git
#       GIT_TAG 3.1.1
#       GIT_PROGRESS ON
#       PREFIX ${BUILD_DIR}
#       BUILD_IN_SOURCE 0
#       BUILD_ALWAYS 0
#       UPDATE_COMMAND ""
#       UPDATE_COMMAND echo add_subdirectory(source) > CMakeLists.txt
#       INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
#       CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#                  -DCMAKE_PROJECT_NAME=x265 -DENABLE_SHARED:bool=ON -DENABLE_PIC:bool=ON
#       BUILD_COMMAND $(MAKE)
# )
ExternalProject_add(libvpx
      GIT_REPOSITORY https://chromium.googlesource.com/webm/libvpx.git
      GIT_TAG v1.9.0
      GIT_PROGRESS ON
      PREFIX libvpx
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND <SOURCE_DIR>/configure
            --prefix=<INSTALL_DIR>
            --enable-static
            --disable-shared
#            --enable-shared
#            --disable-static
            --disable-examples
      BUILD_COMMAND $(MAKE)
)

ExternalProject_Get_Property(libvpx INSTALL_DIR)
set(libvpx_INSTALL_DIR ${INSTALL_DIR})

set(FFMPEG_TARGET ffmpeg)
ExternalProject_add(${FFMPEG_TARGET}
      URL http://ffmpeg.org/releases/ffmpeg-4.3.1.tar.bz2
      URL_HASH MD5=804707549590e90880e8ecd4e5244fd8
      DOWNLOAD_DIR ${BUILD_DIR}/download/ffmpeg
      PREFIX ${FFMPEG_TARGET}
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND <SOURCE_DIR>/configure
            --prefix=<INSTALL_DIR>
            --extra-cflags="-I${libvpx_INSTALL_DIR}/include"
            --extra-ldflags="-L${libvpx_INSTALL_DIR}/lib"
            --enable-static
            --disable-shared
#            --enable-shared
#            --disable-static
            --disable-gpl
            --enable-nonfree
            # --enable-libfreetype
            # --enable-libfdk-aac # audio
            # --enable-libmp3lame # audio
            # --enable-libopus # audio
            # --enable-libvorbis # audio
            --enable-libvpx
            # --enable-libx264 # gpl
            # --enable-libx265 # gpl
      BUILD_COMMAND $(MAKE)
      DEPENDS libvpx
)

ExternalProject_Get_Property(${FFMPEG_TARGET} INSTALL_DIR)

set(FFMPEG_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(FFMPEG_LIB_DIR ${INSTALL_DIR}/lib)
set(FFMPEG_LIBRARIES ${FFMPEG_TARGET})
endif()

# Add OpenImageIO
set(OPENIMAGEIO_TARGET openimageio)
ExternalProject_Add(${OPENIMAGEIO_TARGET}
      URL https://github.com/OpenImageIO/oiio/archive/Release-2.2.11.1.tar.gz
      URL_HASH MD5=43eb3e6cc6ca1cbfd55bbb2f19688c95
      DOWNLOAD_DIR ${BUILD_DIR}/download/oiio
      PREFIX ${OPENIMAGEIO_TARGET}
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      #INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
      CONFIGURE_COMMAND ${CMAKE_COMMAND} ${CMAKE_CORE_BUILD_FLAGS}
            -DCMAKE_PREFIX_PATH=${CMAKE_INSTALL_PREFIX}
            -DBOOST_ROOT=${BOOST_INCLUDE_DIRS} -DOIIO_BUILD_TESTS:BOOL=OFF
            -DILMBASE_HOME="${OPENEXR_LIB_DIR}/../"
            -DOPENEXR_HOME="${OPENEXR_LIB_DIR}/../"
            ${TIFF_CMAKE_FLAGS} ${ZLIB_CMAKE_FLAGS} ${PNG_CMAKE_FLAGS} ${JPEG_CMAKE_FLAGS} ${LIBRAW_CMAKE_FLAGS} ${OPENEXR_CMAKE_FLAGS}
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
            -DSTOP_ON_WARNING=OFF
            -DUSE_FFMPEG=${AV_BUILD_FFMPEG}
            -DUSE_TURBOJPEG=${AV_BUILD_JPEG}
            -DUSE_LIBRAW=${AV_BUILD_LIBRAW}
            -DUSE_OPENEXR=${AV_BUILD_OPENEXR}
            -DUSE_TIFF=${AV_BUILD_TIFF}
            -DUSE_PNG=ON
            -DUSE_PYTHON=OFF -DUSE_OPENCV=OFF -DUSE_OPENGL=OFF
            # TODO: build with libheif
      BUILD_COMMAND $(MAKE)
      DEPENDS   ${BOOST_TARGET}
                ${OPENEXR_TARGET}
                ${TIFF_TARGET}
                ${PNG_TARGET}
                ${JPEG_TARGET}
                ${LIBRAW_TARGET}
                ${ZLIB_TARGET}
                ${FFMPEG_TARGET}
      )

ExternalProject_Get_Property(${OPENIMAGEIO_TARGET} INSTALL_DIR)
# TODO: openjpeg
# -DOPENJPEG_INCLUDE_DIR=$OPENJPEG_INCLUDE_DIR/openjpeg-2.0 -DOPENJPEG_OPENJP2_LIBRARIES=$OPENJPEG_OPENJP2_LIBRARIES
set(OPENIMAGEIO_CMAKE_FLAGS -DOpenImageIO_DIR=${INSTALL_DIR})
set(OPENIMAGEIO_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(OPENIMAGEIO_LIB_DIR ${INSTALL_DIR}/lib)
set(OPENIMAGEIO_LIBRARIES ${OPENIMAGEIO_TARGET} ${OPENIMAGEIO_TARGET}_Util)

## Add Alembic: I/O for Point Cloud and Cameras
set(ALEMBIC_TARGET alembic)
ExternalProject_Add(${ALEMBIC_TARGET}
      # vfxplatform CY2020 1.7.x
      URL https://github.com/alembic/alembic/archive/1.7.16.tar.gz
      URL_HASH MD5=effcc86e42fe6605588e3de57bde6677
      DOWNLOAD_DIR ${BUILD_DIR}/download/alembic
      PREFIX ${ALEMBIC_TARGET}
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ${CMAKE_COMMAND} ${CMAKE_CORE_BUILD_FLAGS}
                          ${ZLIB_CMAKE_FLAGS}
                          ${ILMBASE_CMAKE_FLAGS}
                          -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
      BUILD_COMMAND $(MAKE)
      DEPENDS ${BOOST_TARGET} ${OPENEXR_TARGET} ${ZLIB_TARGET}
      )

ExternalProject_Get_Property(${ALEMBIC_TARGET} INSTALL_DIR)
set(ALEMBIC_CMAKE_FLAGS -DAlembic_DIR:PATH=${INSTALL_DIR}/lib/cmake/Alembic)
set(ALEMBIC_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(ALEMBIC_LIB_DIR ${INSTALL_DIR}/lib)
set(ALEMBIC_LIBRARIES ${ALEMBIC_TARGET})

if(AV_BUILD_OPENCV)
set(OPENCV_TARGET opencv)
ExternalProject_Add(opencv_contrib
  URL https://github.com/opencv/opencv_contrib/archive/4.5.1.zip
  URL_MD5 ddb4f64d6cf31d589a8104655d39c99b
  DOWNLOAD_DIR ${BUILD_DIR}/download/opencv_contrib
  PREFIX opencv_contrib
  BUILD_ALWAYS 0
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property(opencv_contrib SOURCE_DIR)
set(opencv_contrib_SOURCE_DIR ${SOURCE_DIR})

ExternalProject_Add(${OPENCV_TARGET}
  URL https://github.com/opencv/opencv/archive/4.5.1.zip
  URL_MD5 cc13d83c3bf989b0487bb3798375ee08
  DOWNLOAD_DIR ${BUILD_DIR}/download/opencv
  PREFIX ${OPENCV_TARGET}
  UPDATE_COMMAND ""
  BUILD_IN_SOURCE 0
  BUILD_ALWAYS 0
  CONFIGURE_COMMAND ${CMAKE_COMMAND} ${CMAKE_CORE_BUILD_FLAGS}
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
    -DOPENCV_EXTRA_MODULES_PATH=${opencv_contrib_SOURCE_DIR}/modules
    ${ZLIB_CMAKE_FLAGS} ${TBB_CMAKE_FLAGS}
    ${TIFF_CMAKE_FLAGS} ${PNG_CMAKE_FLAGS}
    ${JPEG_CMAKE_FLAGS} ${LIBRAW_CMAKE_FLAGS}
    -DWITH_TBB=ON
    -DWITH_FFMPEG=${AV_BUILD_FFMPEG}
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
    -DBUILD_SHARED_LIBS=ON
    -DWITH_CUDA=OFF
    -DWITH_OPENCL=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_LIST=core,improc,photo,objdetect,video,imgcodecs,videoio,features2d,xfeatures2d,version
    <SOURCE_DIR>
  BUILD_COMMAND $(MAKE)

  DEPENDS opencv_contrib
          ${TBB_TARGET}
          ${ZLIB_TARGET}
          ${OPENEXR_TARGET}
          ${TIFF_TARGET}
          ${PNG_TARGET}
          ${JPEG_TARGET}
          ${LIBRAW_TARGET}
          ${FFMPEG_TARGET}
)
# set(OPENCV_CMAKE_FLAGS -DOpenCV_DIR=${BUILD_DIR}/opencv_install -DCMAKE_PREFIX_PATH=${BUILD_DIR}/opencv_install)
ExternalProject_Get_Property(${OPENCV_TARGET} INSTALL_DIR)
set(OPENCV_CMAKE_FLAGS -DOpenCV_DIR=${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/opencv4 -DOPENCV_DIR=${INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/opencv4)
set(OPENCV_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(OPENCV_LIB_DIR ${INSTALL_DIR}/lib)
set(OPENCV_LIBRARIES ${OPENCV_TARGET})
endif()

# Add CCTag
if(AV_BUILD_CCTAG)
set(CCTAG_TARGET cctag)
ExternalProject_Add(${CCTAG_TARGET}
      GIT_REPOSITORY https://github.com/alicevision/CCTag
      # GIT_TAG boost-no-cxx11-constexpr
      GIT_TAG ba0daba0ff1e2c4e2698220ab6ccfc06e5ede589
      PREFIX ${CCTAG_TARGET}
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ${CMAKE_COMMAND} ${CMAKE_CORE_BUILD_FLAGS}
      ${BOOST_CMAKE_FLAGS}
      ${OPENCV_CMAKE_FLAGS}
      ${EIGEN_CMAKE_FLAGS}
      ${TBB_CMAKE_FLAGS}
      ${CUDA_CMAKE_FLAGS}
      -DCMAKE_CUDA_FLAGS=${CUDA_GENCODES}
      -DCCTAG_WITH_CUDA:BOOL=OFF # TODO: ${BUILD_CUDA_MODULE}
      -DCCTAG_BUILD_TESTS=OFF
      -DCCTAG_BUILD_APPS=OFF
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
      BUILD_COMMAND $(MAKE)
      DEPENDS ${BOOST_TARGET} ${OPENCV_TARGET} ${EIGEN3_TARGET} ${TBB_TARGET}
      )

ExternalProject_Get_Property(${CCTAG_TARGET} INSTALL_DIR)
set(CCTAG_CMAKE_FLAGS -DCCTag_DIR:PATH=${INSTALL_DIR}/lib/cmake/CCTag)
set(CCTAG_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(CCTAG_LIB_DIR ${INSTALL_DIR}/lib)
set(CCTAG_LIBRARIES ${CCTAG_TARGET}$<$<CONFIG:Debug>:d>)
endif()

# Add PopSift
if(AV_BUILD_POPSIFT)
set(POPSIFT_TARGET popsift)
ExternalProject_Add(${POPSIFT_TARGET}
      GIT_REPOSITORY https://github.com/alicevision/popsift
      GIT_TAG 5bbd332f94a280535d54928ced9c3fb74f16a3fb  #v1.0.0-rc3
      PREFIX ${POPSIFT_TARGET}
      BUILD_IN_SOURCE 0
      BUILD_ALWAYS 0
      UPDATE_COMMAND ""
      CONFIGURE_COMMAND ${CMAKE_COMMAND} ${CMAKE_CORE_BUILD_FLAGS}
              ${BOOST_CMAKE_FLAGS}
              ${CUDA_CMAKE_FLAGS}
              -DPopSift_BUILD_EXAMPLES:BOOL=OFF
              -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
      BUILD_COMMAND $(MAKE)
      DEPENDS ${BOOST_TARGET}
      )

ExternalProject_Get_Property(${POPSIFT_TARGET} INSTALL_DIR)
set(POPSIFT_CMAKE_FLAGS -DPopSift_DIR:PATH=${INSTALL_DIR}/lib/cmake/PopSift)
set(POPSIFT_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(POPSIFT_LIB_DIR ${INSTALL_DIR}/lib)
set(POPSIFT_LIBRARIES ${POPSIFT_TARGET}$<$<CONFIG:Debug>:d>)
endif()


set(AV_INCLUDE_DIRS
  ${GEOGRAM_INCLUDE_DIRS}
  ${OPENGV_INCLUDE_DIRS}
  ${LAPACK_INCLUDE_DIRS}
  ${SUITESPARSE_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${OPENEXR_INCLUDE_DIRS}
  ${TIFF_INCLUDE_DIRS}
  ${LIBRAW_INCLUDE_DIRS}
  ${FFMPEG_INCLUDE_DIRS}
  ${OPENIMAGEIO_INCLUDE_DIRS}
  ${ALEMBIC_INCLUDE_DIRS}
  ${OPENCV_INCLUDE_DIRS}
  ${CCTAG_INCLUDE_DIRS}
  ${POPSIFT_INCLUDE_DIRS}
)

set(AV_LIB_DIRS
  ${GEOGRAM_LIB_DIR}
  ${OPENGV_LIB_DIR}
  ${LAPACK_LIB_DIR}
  ${SUITESPARSE_LIB_DIR}
  ${CERES_LIB_DIR}
  ${OPENEXR_LIB_DIR}
  ${TIFF_LIB_DIR}
  ${LIBRAW_LIB_DIR}
  ${FFMPEG_LIB_DIR}
  ${OPENIMAGEIO_LIB_DIR}
  ${ALEMBIC_LIB_DIR}
  ${OPENCV_LIB_DIR}
  ${CCTAG_LIB_DIR}
  ${POPSIFT_LIB_DIR}
)

set(AV_DEPS_LIBRARIES
  ${GEOGRAM_LIBRARIES}
  ${OPENGV_LIBRARIES}
  ${LAPACKBLAS_LIBRARIES}
  ${SUITESPARSE_LIBRARIES}
  ${CERES_LIBRARIES}
  ${OPENEXR_LIBRARIES}
  ${TIFF_LIBRARIES}
  ${LIBRAW_LIBRARIES}
  ${FFMPEG_LIBRARIES}
  ${OPENIMAGEIO_LIBRARIES}
  ${ALEMBIC_LIBRARIES}
  ${OPENCV_LIBRARIES}
  ${CCTAG_LIBRARIES}
  ${POPSIFT_LIBRARIES}
)

set(AV_DEPS_TARGETS
  ${GEOGRAM_TARGET}
  ${OPENGV_TARGET}
  ${LAPACK_TARGET}
  ${SUITESPARSE_TARGET}
  ${CERES_TARGET}
  ${OPENEXR_TARGET}
  ${TIFF_TARGET}
  ${LIBRAW_TARGET}
  ${FFMPEG_TARGET}
  ${OPENIMAGEIO_TARGET}
  ${ALEMBIC_TARGET}
  ${OPENCV_TARGET}
  ${CCTAG_TARGET}
  ${POPSIFT_TARGET}
)


#message("AV_INCLUDE_DIRS: " ${AV_INCLUDE_DIRS})
#message("AV_LIB_DIRS: " ${AV_LIB_DIRS})
#message("AV_DEPS_LIBRARIES: " ${AV_DEPS_LIBRARIES})
#message("AV_DEPS_TARGETS: " ${AV_DEPS_TARGETS})
#list(GET AV_DEPS_LIBRARIES 1 temp1)
#list(GET AV_DEPS_LIBRARIES 2 temp2)
#list(GET AV_DEPS_LIBRARIES 3 temp3)
#message("temp1: " ${temp1})
#message("temp2: " ${temp2})
#message("temp3: " ${temp3})
#foreach(AV_INFO IN ZIP_LISTS  AV_INCLUDE_DIRS AV_LIB_DIRS AV_DEPS_LIBRARIES AV_DEPS_TARGETS)
#        message("AV TARGET: " ${AV_INFO_3})
#        set(BUILD_SHARED_LIBS_BACK ${BUILD_SHARED_LIBS})
#        set(av_3rdparty_name "3rdparty_${AV_INFO_3}")
#        if (${av_3rdparty_name} STREQUAL "3rdparty_suitesparse")
#            set(BUILD_SHARED_LIBS ON)
#            import_shared_3rdparty_library(${av_3rdparty_name}
#                INCLUDE_DIRS ${AV_INFO_0}
#                LIB_DIR      ${AV_INFO_1}
#                LIBRARIES    ${AV_INFO_2})
#        else()
#            import_3rdparty_library(${av_3rdparty_name}
#                INCLUDE_DIRS ${AV_INFO_0}
#                LIB_DIR      ${AV_INFO_1}
#                LIBRARIES    ${AV_INFO_2}
#            )
#        endif()

#        set(ALICEVISION_TARGET ${av_3rdparty_name})
#        message("av_3rdparty_name: " ${av_3rdparty_name})
#        message("AV_INFO_2: " ${AV_INFO_2})

#        add_dependencies("${av_3rdparty_name}" ${AV_INFO_3})

#        # Putting libs somehow works for Ubuntu.
#        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS "${ALICEVISION_TARGET}")
#        set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACK})
#endforeach()

#set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACK})
#message("BUILD_SHARED_LIBS: " ${BUILD_SHARED_LIBS})

set(BUILD_SHARED_LIBS_BACK ${BUILD_SHARED_LIBS})
function(import_av_libraries AV_INFO_0 AV_INFO_1 AV_INFO_2 AV_INFO_3)
        message("AV TARGET: " ${AV_INFO_3})
        set(BUILD_SHARED_LIBS_BACK ${BUILD_SHARED_LIBS})
        set(av_3rdparty_name "3rdparty_${AV_INFO_3}")
        if (${av_3rdparty_name} STREQUAL "3rdparty_suitesparse")
            set(BUILD_SHARED_LIBS ON)
            import_shared_3rdparty_library(${av_3rdparty_name}
                INCLUDE_DIRS ${AV_INFO_0}
                LIB_DIR      ${AV_INFO_1}
                LIBRARIES    ${AV_INFO_2})
        else()
            import_3rdparty_library(${av_3rdparty_name}
                INCLUDE_DIRS ${AV_INFO_0}
                LIB_DIR      ${AV_INFO_1}
                LIBRARIES    ${AV_INFO_2}
            )
        endif()

        set(ALICEVISION_TARGET ${av_3rdparty_name})
        message("av_3rdparty_name: " ${av_3rdparty_name})
        message("AV_INFO_2: " ${AV_INFO_2})

        add_dependencies("${av_3rdparty_name}" ${AV_INFO_3})

        # Putting libs somehow works for Ubuntu.
        list(APPEND CloudViewer_3RDPARTY_PRIVATE_TARGETS "${ALICEVISION_TARGET}")
        set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACK})
endfunction()

import_av_libraries("${GEOGRAM_INCLUDE_DIRS}" "${GEOGRAM_LIB_DIR}" "${GEOGRAM_LIBRARIES}" "${GEOGRAM_TARGET}")
import_av_libraries("${OPENGV_INCLUDE_DIRS}" "${OPENGV_LIB_DIR}" "${OPENGV_LIBRARIES}" "${OPENGV_TARGET}")
import_av_libraries("${LAPACK_INCLUDE_DIRS}" "${LAPACK_LIB_DIR}" "${LAPACKBLAS_LIBRARIES}" "${LAPACK_TARGET}")
import_av_libraries("${SUITESPARSE_INCLUDE_DIRS}" "${SUITESPARSE_LIB_DIR}" "${SUITESPARSE_LIBRARIES}" "${SUITESPARSE_TARGET}")
import_av_libraries("${CERES_INCLUDE_DIRS}" "${CERES_LIB_DIR}" "${CERES_LIBRARIES}" "${CERES_TARGET}")
import_av_libraries("${OPENEXR_INCLUDE_DIRS}" "${OPENEXR_LIB_DIR}" "${OPENEXR_LIBRARIES}" "${OPENEXR_TARGET}")
import_av_libraries("${TIFF_INCLUDE_DIRS}" "${TIFF_LIB_DIR}" "${TIFF_LIBRARIES}" "${TIFF_TARGET}")
import_av_libraries("${LIBRAW_INCLUDE_DIRS}" "${LIBRAW_LIB_DIR}" "${LIBRAW_LIBRARIES}" "${LIBRAW_TARGET}")
import_av_libraries("${FFMPEG_INCLUDE_DIRS}" "${FFMPEG_LIB_DIR}" "${FFMPEG_LIBRARIES}" "${FFMPEG_TARGET}")
import_av_libraries("${OPENIMAGEIO_INCLUDE_DIRS}" "${OPENIMAGEIO_LIB_DIR}" "${OPENIMAGEIO_LIBRARIES}" "${OPENIMAGEIO_TARGET}")
import_av_libraries("${ALEMBIC_INCLUDE_DIRS}" "${ALEMBIC_LIB_DIR}" "${ALEMBIC_LIBRARIES}" "${ALEMBIC_TARGET}")
import_av_libraries("${OPENCV_INCLUDE_DIRS}" "${OPENCV_LIB_DIR}" "${OPENCV_LIBRARIES}" "${OPENCV_TARGET}")
import_av_libraries("${CCTAG_INCLUDE_DIRS}" "${CCTAG_LIB_DIR}" "${CCTAG_LIBRARIES}" "${CCTAG_TARGET}")
import_av_libraries("${POPSIFT_INCLUDE_DIRS}" "${POPSIFT_LIB_DIR}" "${POPSIFT_LIBRARIES}" "${POPSIFT_TARGET}")

set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_BACK})
