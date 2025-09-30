include(ExternalProject)

# Suppress CMP0135 warning
if (POLICY CMP0135)
    cmake_policy(SET CMP0135 OLD)
endif()

set(lib_name dlib)

ExternalProject_Add(ext_dlib
    PREFIX dlib
    URL https://github.com/davisking/dlib/archive/refs/tags/v19.22.tar.gz
    URL_HASH SHA256=5f44b67f762691b92f3e41dcf9c95dd0f4525b59cacb478094e511fdacb5c096
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/dlib"
    UPDATE_COMMAND ""
    # Fix dlib CMake version issues with cmake 4.x.
    PATCH_COMMAND ${CMAKE_COMMAND} -DINPUT_DIR=<SOURCE_DIR> -P ${CMAKE_CURRENT_LIST_DIR}/fix_all_cmake_minimum.cmake
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.10
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
    BUILD_BYPRODUCTS
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}${CMAKE_STATIC_LIBRARY_SUFFIX}
        <INSTALL_DIR>/lib/${CMAKE_STATIC_LIBRARY_PREFIX}${lib_name}d${CMAKE_STATIC_LIBRARY_SUFFIX})

ExternalProject_Get_Property(ext_dlib SOURCE_DIR)
ExternalProject_Get_Property(ext_dlib INSTALL_DIR)
set(DLIB_ROOT_DIR ${SOURCE_DIR})
set(DLIB_INCLUDE_DIRS ${INSTALL_DIR}/include/) # "/" is critical.
set(DLIB_LIB_DIR ${INSTALL_DIR}/lib)
if (MSVC)
	math(EXPR numbits ${CMAKE_SIZEOF_VOID_P}*8)
	set(DLIB_LIBRARIES ${lib_name}19.22.0_$<IF:$<CONFIG:Debug>,debug,release>_${numbits}bit_msvc${MSVC_VERSION})
else()
	set(DLIB_LIBRARIES ${lib_name})
endif()
