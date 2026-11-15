include(ExternalProject)

# Ceres 2.2 is the first pinned release with a CUDA dense linear algebra
# backend, used by reconstruction's Ceres-vs-Caspar reference gate. It is
# DISABLED BY DEFAULT and must stay that way for distributed artifacts:
# enabling it gives libceres hard DT_NEEDED entries on CUDA runtime
# libraries (libcudart/libcublas/libcusolver/libcusparse), which
# scripts/platforms/linux/pack_ubuntu.sh excludes by design (see the
# VerifyNoDynamicCuda contract) and which lock every host to one CUDA
# toolkit major version. GPU Bundle Adjustment ships through the
# Symforce-Caspar backend instead, whose runtime path is the only supported
# CUDA solver for reconstruction. Flip CERES_ENABLE_CUDA=ON solely for
# developer/CI gates that carry their own CUDA runtime closure, or once a
# static cudart/cublas/cusolver/cusparse link route is adopted.
option(CERES_ENABLE_CUDA
       "Enable the Ceres CUDA dense linear algebra backend (adds hard CUDA \
runtime dependencies to libceres; see the comment above)" OFF)

set(CERES_URL https://github.com/ceres-solver/ceres-solver/archive/refs/tags/2.2.0.tar.gz)
set(CERES_URL_HASH 12efacfadbfdc1bbfa203c236e96f4d3c210bed96994288b3ff0c8e7c6f350d4)

# CUDA-only ExternalProject inputs, assembled here so the default OFF path
# passes no CUDA arguments at all. ExternalProject receives CMAKE_ARGS as a
# CMake list; the string(REPLACE ...) preserves a multi-arch CUDA value as
# one argument rather than splitting it at every semicolon.
set(CERES_CUDA_ARGS "")
set(_CERES_PATCH_COMMAND
        ${CMAKE_COMMAND} -E copy
        ${CloudViewer_3RDPARTY_DIR}/ceres-solver/FindTBB.cmake <SOURCE_DIR>/cmake)
if (CERES_ENABLE_CUDA)
    string(REPLACE ";" "$<SEMICOLON>" CERES_CUDA_ARCHITECTURES_ESCAPED
           "${CMAKE_CUDA_ARCHITECTURES}")
    list(APPEND CERES_CUDA_ARGS
            -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
            -DCMAKE_CUDA_ARCHITECTURES=${CERES_CUDA_ARCHITECTURES_ESCAPED}
            -DCERES_CUDA_ARCHITECTURES=${CERES_CUDA_ARCHITECTURES_ESCAPED}
            -DCMAKE_CUDA_STANDARD=17)
    # Keep the parent reconstruction build's architecture list so the Ceres
    # reference solver and Caspar use the same GPU code targets.
    list(APPEND _CERES_PATCH_COMMAND
            COMMAND ${CMAKE_COMMAND}
                    -DSOURCE_DIR=<SOURCE_DIR>
                    -P ${CloudViewer_3RDPARTY_DIR}/ceres-solver/patch_respect_cuda_architectures.cmake)
endif ()

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
        PATCH_COMMAND ${_CERES_PATCH_COMMAND}
        CMAKE_ARGS
            -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            ${EIGEN_CMAKE_FLAGS}
            ${LAPACK_CMAKE_FLAGS}
            ${GLOG_CMAKE_FLAGS}
            ${GFLAGS_CMAKE_FLAGS}
            ${SUITESPARSE_CMAKE_FLAGS}
            -DBUILD_SHARED_LIBS=$<$<PLATFORM_ID:Linux>:ON:OFF>
            # Installed shared ceres must resolve its own NEEDED glog/gflags/lapack:
            # consumer RUNPATHs are non-transitive on Linux.
            -DCMAKE_INSTALL_RPATH=$ORIGIN
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
            # No-op: Ceres 2.2 removed its OpenMP option entirely (C++11
            # threads only, find_package(Threads) in internal/ceres). Kept so
            # both ExternalProjects see the same WITH_OPENMP knob.
            -DOPENMP=${WITH_OPENMP}
            -DUSE_CUDA=${CERES_ENABLE_CUDA}
            ${CERES_CUDA_ARGS}
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
