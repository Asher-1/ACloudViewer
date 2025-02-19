include(ExternalProject)

if (${GLIBCXX_USE_CXX11_ABI})
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 1)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for gflags")
else ()
    set(CUSTOM_GLIBCXX_USE_CXX11_ABI 0)
    message(STATUS "add -D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI} support for gflags")
endif ()

set(GFLAGS_MAJOR_VERSION "2")
set(GFLAGS_MINOR_VERSION "2.2")
set(GFLAGS_FULL_VERSION ${GFLAGS_MAJOR_VERSION}.${GFLAGS_MINOR_VERSION})

# Add gflags
ExternalProject_Add(
        ext_gflags
        PREFIX gflags
        URL https://github.com/gflags/gflags/archive/v${GFLAGS_FULL_VERSION}.zip
        URL_HASH MD5=ff856ff64757f1381f7da260f79ba79b
        DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/gflags"
        BUILD_IN_SOURCE 0
        BUILD_ALWAYS 0
        INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
        UPDATE_COMMAND ""
        CMAKE_ARGS
            # Syncing GLIBCXX_USE_CXX11_ABI for MSVC causes problems, but directly
            # checking CXX_COMPILER_ID is not supported.
            $<IF:$<PLATFORM_ID:Windows>,"",-DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${CUSTOM_GLIBCXX_USE_CXX11_ABI}>
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
            -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DBUILD_SHARED_LIBS=$<$<PLATFORM_ID:Linux>:ON:OFF>
            -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
            -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

ExternalProject_Get_Property(ext_gflags INSTALL_DIR)
set(GFLAGS_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(GFLAGS_LIB_DIR ${INSTALL_DIR}/lib)
set(EXT_GFLAGS_LIBRARIES gflags)
set(GFLAGS_CMAKE_FLAGS -Dgflags_DIR=${GFLAGS_LIB_DIR}/cmake/gflags -DGFLAGS_INCLUDE_DIR_HINTS=${GFLAGS_INCLUDE_DIRS} -DGFLAGS_LIBRARY_DIR_HINTS=${GFLAGS_LIB_DIR})
if (MSVC)
    set(EXT_GFLAGS_LIBRARIES gflags_static$<$<CONFIG:Debug>:_debug>)
    set(GFLAGS_CMAKE_FLAGS ${GFLAGS_CMAKE_FLAGS} -DGFLAGS_DLL_DECLARE_FLAG= -DGFLAGS_DLL_DEFINE_FLAG= -DGFLAGS_IS_A_DLL=0)
endif ()

# fix missing libgflags symlinks issues on linux
if (UNIX AND NOT APPLE)
    set(CUSTOM_GFLAGS_LIB_NAME ${CMAKE_SHARED_LIBRARY_PREFIX}${EXT_GFLAGS_LIBRARIES}${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(LINK_GFLAGS_FILENAME ${CUSTOM_GFLAGS_LIB_NAME}.${GFLAGS_MINOR_VERSION})
    ExternalProject_Add_Step(ext_gflags replace_gflags_symlink_with_file
        COMMAND ${CMAKE_COMMAND} -E remove ${LINK_GFLAGS_FILENAME}
        COMMAND ${CMAKE_COMMAND} -E copy
        ${CUSTOM_GFLAGS_LIB_NAME}.${GFLAGS_FULL_VERSION}
        ${LINK_GFLAGS_FILENAME}
        WORKING_DIRECTORY "${GFLAGS_LIB_DIR}"
        DEPENDEES install
    )
    
    # for debugging
    copy_shared_library(ext_gflags
        LIB_DIR ${GFLAGS_LIB_DIR}
        LIBRARIES ${EXT_GFLAGS_LIBRARIES}
        SUFFIX ".${GFLAGS_MINOR_VERSION}"
    )

    # fix missing installed gflags issues on linux
    set(LINK_GFLAGS_FILE_PATH ${GFLAGS_LIB_DIR}/${LINK_GFLAGS_FILENAME})
    cloudViewer_install_ext(FILES ${LINK_GFLAGS_FILE_PATH} ${CMAKE_INSTALL_PREFIX}/${CloudViewer_INSTALL_LIB_DIR} "")
endif()
