include(ExternalProject)
find_package(Git REQUIRED)

# Keep the Bitmap backend's source, ABI, and optional-format set identical on
# every supported platform. In particular, OIIO must never discover a host
# OpenCV: doing so adds OpenCV to the Reconstruction runtime dependency graph.
set(OPENIMAGEIO_VERSION "3.1.17.0")
set(OPENIMAGEIO_SHA256
    "92a26c0af4ffc6676d72d9dfe0e991eb45fdf3192abee3d0855a24d6c721b013")

# The active Conda prefix remains an input for compiler-adjacent prerequisites
# on macOS and Windows. Preserve it as one ExternalProject argument.
string(REPLACE ";" "$<SEMICOLON>" OPENIMAGEIO_PREFIX_PATH
       "${CMAKE_PREFIX_PATH}")

if(WIN32)
    set(_openimageio_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/OpenImageIO.lib")
    set(_openimageio_util_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/OpenImageIO_Util.lib")
else()
    set(_openimageio_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}OpenImageIO${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_openimageio_util_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}OpenImageIO_Util${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()

ExternalProject_Add(ext_openimageio
    PREFIX openimageio-${OPENIMAGEIO_VERSION}
    URL https://github.com/OpenImageIO/oiio/archive/refs/tags/v${OPENIMAGEIO_VERSION}.tar.gz
    URL_HASH SHA256=${OPENIMAGEIO_SHA256}
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/openimageio"
    UPDATE_COMMAND ""
    BUILD_IN_SOURCE OFF
    BUILD_ALWAYS 0
    INSTALL_DIR ${CLOUDVIEWER_EXTERNAL_INSTALL_DIR}
    BUILD_BYPRODUCTS
        ${_openimageio_link}
        ${_openimageio_util_link}
    PATCH_COMMAND ${CMAKE_COMMAND}
        -DSOURCE_DIR=<SOURCE_DIR>
        -DPATCH_FILE=${CMAKE_CURRENT_LIST_DIR}/patches/0001-provide-local-ocio-pystring-to-config.patch
        -DGIT_EXECUTABLE=${GIT_EXECUTABLE}
        -P ${CMAKE_CURRENT_LIST_DIR}/patches/apply_openimageio_patch.cmake
    CMAKE_ARGS
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        ${ExternalProject_CMAKE_ARGS_hidden}
        -DCMAKE_BUILD_TYPE=$<IF:$<PLATFORM_ID:Windows>,${CMAKE_BUILD_TYPE},Release>
        -DCMAKE_PREFIX_PATH=${OPENIMAGEIO_PREFIX_PATH}
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        -DCMAKE_INSTALL_LIBDIR=${CloudViewer_INSTALL_LIB_DIR}
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_TESTING=OFF
        -DOIIO_BUILD_TESTS=OFF
        -DOIIO_BUILD_TOOLS=OFF
        -DBUILD_DOCS=OFF
        -DINSTALL_DOCS=OFF
        -DINSTALL_FONTS=OFF
        -DEMBEDPLUGINS=ON
        -DUSE_PYTHON=OFF
        -DUSE_QT=OFF
        -DUSE_OPENCV=OFF
        -DUSE_FREETYPE=OFF
        -DUSE_TBB=OFF
        -DUSE_FFMPEG=OFF
        -DUSE_GIF=OFF
        -DUSE_LIBHEIF=OFF
        -DUSE_LIBRAW=OFF
        -DUSE_OPENJPEG=OFF
        -DUSE_OPENJPH=OFF
        -DUSE_OPENVDB=OFF
        -DUSE_PTEX=OFF
        -DUSE_WEBP=OFF
        -DUSE_JXL=OFF
        -DOpenImageIO_BUILD_MISSING_DEPS=required
    DEPENDS ext_zlib)

ExternalProject_Get_Property(ext_openimageio INSTALL_DIR)
string(REPLACE "<INSTALL_DIR>" "${INSTALL_DIR}" _openimageio_link
       "${_openimageio_link}")
string(REPLACE "<INSTALL_DIR>" "${INSTALL_DIR}" _openimageio_util_link
       "${_openimageio_util_link}")

# Reconstruction has no OIIO types in its public ABI. Expose only an interface
# target and keep OIIO's internal fmt headers out of unrelated targets.
add_library(3rdparty_openimageio INTERFACE)
target_include_directories(3rdparty_openimageio SYSTEM INTERFACE
    "$<BUILD_INTERFACE:${INSTALL_DIR}/include>")
target_link_libraries(3rdparty_openimageio INTERFACE
    "$<BUILD_INTERFACE:${_openimageio_link}>"
    "$<BUILD_INTERFACE:${_openimageio_util_link}>")
add_dependencies(3rdparty_openimageio ext_openimageio)

set(OPENIMAGEIO_TARGET 3rdparty_openimageio)
message(STATUS
    "Reconstruction image backend: OpenImageIO ${OPENIMAGEIO_VERSION} from 3rdparty source (OpenCV disabled)")
