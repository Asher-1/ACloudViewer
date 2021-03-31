include(ExternalProject)

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/lapack-3.9.0.tar.gz"
    REMOTE_URLS "https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz"
)

ExternalProject_Add(
       ext_lapack
       #   http://www.netlib.org/lapack/lapack-3.9.0.tar.gz
       URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
       URL_HASH MD5=0b251e2a8d5f949f99b50dd5e2200ee2
       PREFIX lapack
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       UPDATE_COMMAND ""
       CMAKE_ARGS
           -DBUILD_SHARED_LIBS=OFF
           -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
           -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
           -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
           -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
       DEPENDS ${TBB_TARGET}
       )

ExternalProject_Get_Property(ext_lapack INSTALL_DIR)
set(BLAS_LIBRARIES ${INSTALL_DIR}/lib/libblas.a)
set(LAPACK_LIBRARIES ${INSTALL_DIR}/lib/liblapack.a)
set(LAPACK_CMAKE_FLAGS -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES})

set(LAPACK_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(LAPACK_LIB_DIR ${INSTALL_DIR}/lib)
set(LAPACKBLAS_LIBRARIES lapack blas)
