include(ExternalProject)

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/gflags-2.2.2.zip"
    REMOTE_URLS "https://github.com/gflags/gflags/archive/v2.2.2.zip"
)

# Add gflags
ExternalProject_Add(
    ext_gflags
    PREFIX ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}
    URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
    URL_HASH MD5=ff856ff64757f1381f7da260f79ba79b
    DOWNLOAD_DIR ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}/download/gflags
    BUILD_IN_SOURCE 0
    BUILD_ALWAYS 0
    UPDATE_COMMAND ""
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/gflags
    BINARY_DIR ${CLOUDVIEWER_EXTERNAL_BUILD_DIR}/gflags_build
    INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
    CMAKE_ARGS
        -DBUILD_SHARED_LIBS=$<IF:$<PLATFORM_ID:Linux>,ON,OFF>
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Get_Property(ext_gflags INSTALL_DIR)
set(GFLAGS_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GFLAGS_LIB_DIR ${INSTALL_DIR}/lib)
if(MSVC)
    set(EXT_GFLAGS_LIBRARIES gflags_static$<$<CONFIG:Debug>:_debug>)
else()
    set(EXT_GFLAGS_LIBRARIES gflags)
endif()
set(GFLAGS_CMAKE_FLAGS -Dgflags_DIR=${GFLAGS_LIB_DIR}/cmake/gflags)
