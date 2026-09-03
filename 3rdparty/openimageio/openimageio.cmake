# OpenImageIO is supplied by the active Conda environment on macOS/Windows and
# by the platform package manager on Ubuntu. Keep the native target private to
# this adapter: OpenImageIO exports an embedded fmt include directory which
# must not enter CloudViewer's global third-party include graph.
find_package(OpenImageIO CONFIG QUIET)

if (TARGET OpenImageIO::OpenImageIO)
    set(_openimageio_native_target OpenImageIO::OpenImageIO)
elseif (TARGET OpenImageIO)
    set(_openimageio_native_target OpenImageIO)
else ()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(OPENIMAGEIO REQUIRED IMPORTED_TARGET OpenImageIO)
    set(_openimageio_native_target PkgConfig::OPENIMAGEIO)
endif ()

add_library(3rdparty_openimageio INTERFACE)
target_link_libraries(3rdparty_openimageio INTERFACE ${_openimageio_native_target})
set(OPENIMAGEIO_TARGET 3rdparty_openimageio)

message(STATUS "Reconstruction image backend: ${_openimageio_native_target}")
