include(ExternalProject)

set(FMT_LIB_NAME fmt)

if (WIN32 AND BUILD_CUDA_MODULE)
    set(FMT_VER "10.1.1")
    set(FMT_SHA256 "78b8c0a72b1c35e4443a7e308df52498252d1cefc2b08c9a97bc9ee6cfe61f8b")
else ()
    set(FMT_VER "10.2.1")
    set(FMT_SHA256 "1250e4cc58bf06ee631567523f48848dc4596133e163f02615c97f78bab6c811")
endif()

ExternalProject_Add(ext_fmt
    PREFIX fmt
    URL https://github.com/fmtlib/fmt/archive/refs/tags/${FMT_VER}.tar.gz
    URL_HASH SHA256=${FMT_SHA256}
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/fmt"
    UPDATE_COMMAND ""
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DFMT_DOC=OFF
        -DFMT_TEST=OFF
        -DFMT_FUZZ=OFF
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${FMT_LIB_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}${FMT_LIB_NAME}d${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Get_Property(ext_fmt INSTALL_DIR)
set(FMT_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(FMT_LIB_DIR ${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR})
# set(FMT_LIBRARIES ${FMT_LIB_NAME}$<$<PLATFORM_ID:Windows>:$<$<CONFIG:Debug>:d>>)
set(FMT_LIBRARIES ${FMT_LIB_NAME}$<$<CONFIG:Debug>:d>)

if (MSVC) # error C2027: undefined type “std::locale” when local building
    ExternalProject_Add_Step(ext_fmt remove_fmt_locale_header
        COMMAND ${CMAKE_COMMAND} -E echo "Checking if locale.h exists..."
        COMMAND ${CMAKE_COMMAND} -E cmake_echo_color --cyan "Checking if locale.h exists..."
        COMMAND ${CMAKE_COMMAND} -E remove -f "${FMT_INCLUDE_DIRS}/locale.h"
        COMMENT "Removing locale.h if it exists"
        WORKING_DIRECTORY "${FMT_INCLUDE_DIRS}"
        DEPENDEES install
        )
endif()