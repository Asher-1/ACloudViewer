include(ExternalProject)

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/glog-0.3.5.zip"
    REMOTE_URLS "https://github.com/google/glog/archive/v0.3.5.zip"
)

ExternalProject_Add(
    ext_glog
    PREFIX glog
    URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
    URL_HASH MD5=454766d0124951091c95bad33dafeacd
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE ON
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -Dgflags_DIR=${GFLAGS_LIB_DIR}/cmake/gflags
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
    BUILD_COMMAND $(MAKE)
    DEPENDS ${GFLAGS_TARGET}
)

ExternalProject_Get_Property(ext_glog INSTALL_DIR)
set(GLOG_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GLOG_LIB_DIR ${INSTALL_DIR}/lib)
set(GLOG_LIBRARIES glog)
