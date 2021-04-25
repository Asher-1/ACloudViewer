include(ExternalProject)

set_local_or_remote_url(
    DOWNLOAD_URL_PRIMARY
    LOCAL_URL   "${THIRD_PARTY_DOWNLOAD_DIR}/gflags-2.2.2.zip"
    REMOTE_URLS "https://github.com/gflags/gflags/archive/v2.2.2.zip"
)

# Add gflags
ExternalProject_Add(
    ext_gflags
    PREFIX gflags
    URL ${DOWNLOAD_URL_PRIMARY} ${DOWNLOAD_URL_FALLBACK}
    URL_HASH MD5=ff856ff64757f1381f7da260f79ba79b
    UPDATE_COMMAND ""
    CONFIGURE_COMMAND ""
    BUILD_IN_SOURCE ON
    CONFIGURE_COMMAND ${CMAKE_COMMAND}
        -DBUILD_SHARED_LIBS=OFF
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> <SOURCE_DIR>
    BUILD_COMMAND $(MAKE)
)

ExternalProject_Get_Property(ext_gflags INSTALL_DIR)
set(GFLAGS_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GFLAGS_LIB_DIR ${INSTALL_DIR}/lib)
if(MSVC)
    set(GFLAGS_LIBRARIES gflags_static$<$<CONFIG:Debug>:_debug>)
else()
    set(GFLAGS_LIBRARIES gflags$<$<CONFIG:Debug>:_debug)
endif()
