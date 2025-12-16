cmake_minimum_required(VERSION ${CLOUDVIEWER_PLUGIN_CMAKE_MINIMUM_VERSION})

project(QuaZip)

# CMP0042: Explicitly acknowledge MACOSX_RPATH
# (introduced in CMake 2.8.12, enabled by default in CMake 3.0,
# and producing a warning when unset since 3.8.0)
cmake_policy(SET CMP0042 NEW)

# Qt5/Qt6 compatibility - use unified Qt:: prefix
# Note: Core is already found in CMakeExternalLibs.cmake
if(USE_QT6)
    set(QUAZIP_LIB_VERSION_SUFFIX 6)
    macro(qt_wrap_cpp)
        qt6_wrap_cpp(${ARGN})
    endmacro()
else()
    set(QUAZIP_LIB_VERSION_SUFFIX 5)
    macro(qt_wrap_cpp)
        qt5_wrap_cpp(${ARGN})
    endmacro()
endif()

# Use system zlib on unix/mingw and Qt ZLIB on Windows (MSVC)
if(UNIX OR MINGW)
    find_package(ZLIB REQUIRED)
elseif(WIN32 AND MSVC)
    # Try to use Qt's bundled zlib on Windows MSVC (Qt5/Qt6 compatibility)
    # QT_ROOT_PATH is set in CMakeExternalLibs.cmake
    if(DEFINED QT_ROOT_PATH)
        set(ZLIB_INCLUDE_DIRS "${QT_ROOT_PATH}/src/3rdparty/zlib" CACHE STRING "Path to ZLIB headers of Qt")
    endif()
    set(ZLIB_LIBRARIES "")
    if(NOT EXISTS "${ZLIB_INCLUDE_DIRS}/zlib.h")
        message(WARNING "Could not find Qt's bundled zlib at ${ZLIB_INCLUDE_DIRS}. Please specify a valid zlib include dir or install system zlib.")
    endif()
else()
    # Fallback: try to find system zlib
    find_package(ZLIB QUIET)
    if(NOT ZLIB_FOUND)
        message(WARNING "Could not find zlib. Please install zlib or specify ZLIB_INCLUDE_DIRS.")
    endif()
endif()

# All build libraries are moved to this directory
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})

set(LIB_SUFFIX "" CACHE STRING "Define suffix of directory name (32/64)")
set(LIB_DESTINATION "${CMAKE_INSTALL_PREFIX}/lib${LIB_SUFFIX}" CACHE STRING "Library directory name" FORCE)
set(QUAZIP_LIB_TARGET_NAME quazip${QUAZIP_LIB_VERSION_SUFFIX} CACHE
    INTERNAL "Target name of libquazip" FORCE)

add_subdirectory(quazip)

install(FILES FindQuaZip.cmake RENAME FindQuaZip${QUAZIP_LIB_VERSION_SUFFIX}.cmake DESTINATION ${CMAKE_ROOT}/Modules)
