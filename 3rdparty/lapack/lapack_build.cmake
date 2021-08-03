include(ExternalProject)

ExternalProject_Add(
       ext_lapack
       PREFIX lapack
       #   http://www.netlib.org/lapack/lapack-3.9.0.tar.gz
       URL https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz
       URL_HASH MD5=0b251e2a8d5f949f99b50dd5e2200ee2
       DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/lapack"
       BUILD_IN_SOURCE 0
       BUILD_ALWAYS 0
       INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
       UPDATE_COMMAND ""
       CMAKE_ARGS
           -DBUILD_SHARED_LIBS=$<IF:$<PLATFORM_ID:Linux>,ON,OFF>
           -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
           -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
           -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
           -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        DEPENDS ${GLOG_TARGET}
       )

ExternalProject_Get_Property(ext_lapack INSTALL_DIR)
set(BLAS_LIBRARIES ${INSTALL_DIR}/lib/libblas.so)
set(LAPACK_LIBRARIES ${INSTALL_DIR}/lib/liblapack.so)
set(LAPACK_CMAKE_FLAGS -DBLAS=${BLAS_LIBRARIES} -DLAPACK=${LAPACK_LIBRARIES} -DBLAS_LIBRARIES=${BLAS_LIBRARIES} -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES})

set(LAPACK_INCLUDE_DIRS ${INSTALL_DIR}/include/)
set(LAPACK_LIB_DIR ${INSTALL_DIR}/lib)
set(LAPACKBLAS_LIBRARIES lapack blas)

if(NOT WIN32)
    cloudViewer_install_ext( FILES ${LAPACK_LIBRARIES} ${INSTALL_DESTINATIONS} "")
    cloudViewer_install_ext( FILES ${BLAS_LIBRARIES} ${INSTALL_DESTINATIONS} "")
endif()
