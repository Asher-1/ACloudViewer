include(ExternalProject)

if (${GLIBCXX_USE_CXX11_ABI})
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 1)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for ceres")
else ()
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 0)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for ceres")
endif ()

set(CERES_URL https://github.com/ceres-solver/ceres-solver/archive/1.14.0.zip)
set(CERES_URL_HASH 26b255b7a9f330bbc1def3b839724a2a)
# set(CERES_URL https://github.com/ceres-solver/ceres-solver/archive/2.0.0.zip)
# set(CERES_URL_HASH 589ec4b9461476285ae0332f2a74be48)


ExternalProject_Add(
        ext_ceres
        PREFIX ceres
        URL ${CERES_URL}
        URL_HASH MD5=${CERES_URL_HASH}
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ceres"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        # PATCH_COMMAND sed "s/tbb_stddef.h/tbb.h/" -i <SOURCE_DIR>/cmake/FindTBB.cmake
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CloudViewer_3RDPARTY_DIR}/ceres-solver/FindTBB.cmake <SOURCE_DIR>/cmake
        CMAKE_ARGS
        ${EIGEN_CMAKE_FLAGS}
        ${LAPACK_CMAKE_FLAGS}
        ${GLOG_CMAKE_FLAGS}
        ${GFLAGS_CMAKE_FLAGS}
        ${SUITESPARSE_CMAKE_FLAGS}
        -DBUILD_SHARED_LIBS=$<$<PLATFORM_ID:Linux>:ON:OFF>
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        # Syncing GLIBCXX_USE_CXX11_ABI for MSVC causes problems, but directly
        # checking CXX_COMPILER_ID is not supported.
        $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI}>
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DGFLAGS=ON
        -DLAPACK=ON
        -DSUITESPARSE=ON
        -DOPENMP=${WITH_OPENMP}
        -DBUILD_BENCHMARKS=OFF
        -DBUILD_TESTING=OFF
        -DBUILD_EXAMPLES=OFF
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        DEPENDS ${INTERNAL_EIGEN3_TARGET} ${SUITESPARSE_TARGET} ext_gflags ext_glog
)

ExternalProject_Get_Property(ext_ceres INSTALL_DIR)
set(CERES_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CERES_LIB_DIR ${INSTALL_DIR}/lib)
set(EXT_CERES_LIBRARIES ceres)
if (WIN32)
    set(EXT_CERES_LIBRARIES ceres$<$<CONFIG:Debug>:-debug>)
endif ()

set(CERES_CMAKE_FLAGS ${SUITESPARSE_CMAKE_FLAGS} ${EIGEN_CMAKE_FLAGS} ${GLOG_CMAKE_FLAGS} -DCeres_DIR=${CERES_LIB_DIR}/cmake/Ceres)
