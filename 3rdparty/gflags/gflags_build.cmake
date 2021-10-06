include(ExternalProject)

# Add gflags
ExternalProject_Add(
    ext_gflags
    PREFIX gflags
    URL https://github.com/gflags/gflags/archive/v2.2.2.zip
    URL_HASH MD5=ff856ff64757f1381f7da260f79ba79b
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/gflags"
    BUILD_IN_SOURCE 0
    BUILD_ALWAYS 0
    INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DBUILD_SHARED_LIBS=$<IF:$<PLATFORM_ID:Linux>,ON,OFF>
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Get_Property(ext_gflags INSTALL_DIR)
set(GFLAGS_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GFLAGS_LIB_DIR ${INSTALL_DIR}/lib)
set(GFLAGS_CMAKE_FLAGS -Dgflags_DIR=${GFLAGS_LIB_DIR}/cmake/gflags)
if(MSVC)
    set(EXT_GFLAGS_LIBRARIES gflags_static$<$<CONFIG:Debug>:_debug>)
    set(GFLAGS_CMAKE_FLAGS ${GFLAGS_CMAKE_FLAGS} -DGFLAGS_DLL_DECLARE_FLAG= -DGFLAGS_DLL_DEFINE_FLAG= -DGFLAGS_IS_A_DLL=0)
else()
    set(EXT_GFLAGS_LIBRARIES gflags)
    set(library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${EXT_GFLAGS_LIBRARIES}${CMAKE_SHARED_LIBRARY_SUFFIX})
    cloudViewer_install_ext( FILES ${GFLAGS_LIB_DIR}/${library_filename} ${INSTALL_DESTINATIONS} "")
endif()
