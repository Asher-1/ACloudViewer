include(ExternalProject)

ExternalProject_Add(
        ext_glog
        PREFIX glog
        URL https://github.com/google/glog/archive/v0.3.5.zip
        URL_HASH MD5=454766d0124951091c95bad33dafeacd
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/glog"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        CMAKE_ARGS
        ${GFLAGS_CMAKE_FLAGS}
        -DBUILD_SHARED_LIBS=$<IF:$<PLATFORM_ID:Linux>,ON,OFF>
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
        DEPENDS ${GFLAGS_TARGET}
)

ExternalProject_Get_Property(ext_glog INSTALL_DIR)
set(GLOG_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GLOG_LIB_DIR ${INSTALL_DIR}/lib)
set(EXT_GLOG_LIBRARIES glog)
set(GLOG_CMAKE_FLAGS ${GFLAGS_CMAKE_FLAGS} -Dglog_DIR=${GLOG_LIB_DIR}/cmake/glog -DGLOG_INCLUDE_DIR_HINTS=${GLOG_INCLUDE_DIRS} -DGLOG_LIBRARY_DIR_HINTS=${GLOG_LIB_DIR})

if (MSVC)
    set(GLOG_CMAKE_FLAGS ${GLOG_CMAKE_FLAGS} -DGOOGLE_GLOG_DLL_DECL=)
else ()
    set(library_filename ${CMAKE_SHARED_LIBRARY_PREFIX}${EXT_GLOG_LIBRARIES}${CMAKE_SHARED_LIBRARY_SUFFIX})
    cloudViewer_install_ext(FILES ${GLOG_LIB_DIR}/${library_filename} ${INSTALL_DESTINATIONS} "")
endif ()
