include(ExternalProject)
set(EIGEN_CMAKE_FLAGS ${EIGEN_DISABLE_ALIGN_FLAGS} -DEigen3_DIR:PATH=${EIGEN_ROOT_DIR}/share/eigen3/cmake -DEIGEN3_INCLUDE_DIR=${EIGEN_INCLUDE_DIR} -DEIGEN_INCLUDE_DIR=${EIGEN_INCLUDE_DIR} -DEigen_INCLUDE_DIR=${EIGEN_INCLUDE_DIR})

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/ceres-solver-1.14.0.zip"
    REMOTE_URLS "https://github.com/ceres-solver/ceres-solver/archive/1.14.0.zip"
)

find_package(BLAS)
find_package(LAPACK)
find_package(SuiteSparse)

if (BLAS_FOUND)
    message("-- Found BLAS library: ${BLAS_LIBRARIES}")
endif()
if (LAPACK_FOUND)
    message("-- Found LAPACK library: ${LAPACK_LIBRARIES}")
endif()
if (SUITESPARSE_FOUND)
    message("-- Found SuiteSparse ${SUITESPARSE_VERSION}, building with SuiteSparse.")
endif()

ExternalProject_Add(
       ext_ceres
       PREFIX ceres-solver
       URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
       URL_HASH MD5=26b255b7a9f330bbc1def3b839724a2a
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CMAKE_ARGS
              ${EIGEN_CMAKE_FLAGS}
#              ${SUITESPARSE_CMAKE_FLAGS}
              -DBUILD_SHARED_LIBS=OFF
              -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
              -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
              -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
              -DLAPACK_LIBRARIES={LAPACK_LIBRARIES}
              -DBLAS_LIBRARIES={BLAS_LIBRARIES}
              -DSUITESPARSE_INCLUDE_DIRS={SUITESPARSE_INCLUDE_DIRS}
              -DSUITESPARSE:BOOL={SUITESPARSE_FOUND}
              -DOPENMP=ON
              -DLAPACK:BOOL={LAPACK_FOUND}
              -DBUILD_TESTING:BOOL=OFF
              -DBUILD_EXAMPLES:BOOL=OFF
              -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
       DEPENDS ${EIGEN3_TARGET} ${SUITESPARSE_TARGET}
       )

ExternalProject_Get_Property(ext_ceres INSTALL_DIR)
set(CERES_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(CERES_LIB_DIR ${INSTALL_DIR}/lib)
set(CERES_LIBRARIES ceres$<$<CONFIG:Debug>:-debug>)
