include(ExternalProject)

# Ceres 2.2 is the first pinned release with the CUDA dense linear algebra
# backend used by reconstruction's Ceres-vs-Caspar gate. It remains optional
# and follows BUILD_CUDA_MODULE, so macOS and CPU-only configurations do not
# acquire a CUDA runtime dependency.
set(CERES_URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz)
set(CERES_URL_HASH 12efacfadbfdc1bbfa203c236e96f4d3c210bed96994288b3ff0c8e7c6f350d4)

# ExternalProject receives CMAKE_ARGS as a CMake list. Preserve a multi-arch
# CUDA value as one argument rather than splitting it at every semicolon.
string(REPLACE ";" "$<SEMICOLON>" CERES_CUDA_ARCHITECTURES_ESCAPED
       "${CMAKE_CUDA_ARCHITECTURES}")

ExternalProject_Add(ext_ceres
        # A versioned prefix prevents a configured build directory from
        # reusing Ceres 1.x sources after the pinned CUDA-capable release
        # changes.
        PREFIX ceres-2.2
        URL ${CERES_URL}
        URL_HASH SHA256=${CERES_URL_HASH}
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/ceres"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        # PATCH_COMMAND sed "s/tbb_stddef.h/tbb.h/" -i <SOURCE_DIR>/cmake/FindTBB.cmake
        PATCH_COMMAND ${CMAKE_COMMAND} -E copy ${CloudViewer_3RDPARTY_DIR}/ceres-solver/FindTBB.cmake <SOURCE_DIR>/cmake
        COMMAND ${CMAKE_COMMAND}
                -DSOURCE_DIR=<SOURCE_DIR>
                -P ${CloudViewer_3RDPARTY_DIR}/ceres-solver/patch_respect_cuda_architectures.cmake
        CMAKE_ARGS
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            ${EIGEN_CMAKE_FLAGS}
            ${LAPACK_CMAKE_FLAGS}
            ${GLOG_CMAKE_FLAGS}
            ${GFLAGS_CMAKE_FLAGS}
            ${SUITESPARSE_CMAKE_FLAGS}
            -DBUILD_SHARED_LIBS=$<$<PLATFORM_ID:Linux>:ON:OFF>
            -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DGFLAGS=ON
            # Eigen 3.4's MetisSupport.h assumes 32-bit int idx_t while the
            # bundled SuiteSparse METIS is built with 64-bit idx_t; keep the
            # CHOLMOD path and fall back to AMD ordering for Eigen sparse.
            -DEIGENMETIS=OFF
            -DLAPACK=ON
            -DSUITESPARSE=ON
            -DOPENMP=${WITH_OPENMP}
            -DUSE_CUDA=${BUILD_CUDA_MODULE}
            -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            -DCMAKE_CUDA_ARCHITECTURES=${CERES_CUDA_ARCHITECTURES_ESCAPED}
            -DCERES_CUDA_ARCHITECTURES=${CERES_CUDA_ARCHITECTURES_ESCAPED}
            -DCMAKE_CUDA_STANDARD=17
            -DBUILD_BENCHMARKS=OFF
            -DBUILD_TESTING=OFF
            -DBUILD_EXAMPLES=OFF
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        DEPENDS 3rdparty_eigen3 3rdparty_suitesparse 3rdparty_gflags 3rdparty_glog)

ExternalProject_Get_Property(ext_ceres INSTALL_DIR)
set(CERES_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CERES_LIB_DIR ${INSTALL_DIR}/lib)
set(EXT_CERES_LIBRARIES ceres)
if (WIN32)
    set(EXT_CERES_LIBRARIES ceres$<$<CONFIG:Debug>:-debug>)
endif ()

set(CERES_CMAKE_FLAGS ${SUITESPARSE_CMAKE_FLAGS} ${EIGEN_CMAKE_FLAGS} ${GLOG_CMAKE_FLAGS} -DCeres_DIR=${CERES_LIB_DIR}/cmake/Ceres)
