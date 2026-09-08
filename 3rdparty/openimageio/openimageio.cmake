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

# ext_zlib artifact consumed via ZLIB_LIBRARY below, pinned to the STATIC
# archive on every platform. zlib always builds the static `zlibstatic`
# target; non-MSVC renames it to `z` (libz.a) alongside the shared
# `libz.so`/`libz.dylib`, while MSVC builds with BUILD_SHARED_LIBS=OFF and
# keeps `zlibstatic.lib`. The static pin is required, not stylistic:
# libOpenImageIO ships inside every independently installable component, and
# macOS records a shared zlib dependency in its LC_LOAD_DYLIB as the
# build-tree install name (`libz.1.dylib`/`@rpath/libz.1.dylib`) while the
# payload copy lands under the real file name — the install-name rewrite in
# OpenImageIOPackageRuntime.cmake cannot match that pair, leaving a reference
# into the build tree that verify_oiio_runtime_payload rejects. Embedding
# zlib keeps the OIIO runtime closure to its own two dylibs — the same
# static-only rule the local dep builds enforce (patches/build_ZLIB.cmake
# deletes OIIO's own shared libz for exactly this reason). Never hardcode a
# Linux-only suffix here either: the path is embedded into OIIO's sub-build
# as a hard file prerequisite, so a missing artifact fails it with
# "No rule to make target" (make) / LNK1181 (MSVC).
if(MSVC)
    set(_openimageio_zlib_library "${CMAKE_BINARY_DIR}/zlib/lib/zlibstatic.lib")
else()
    set(_openimageio_zlib_library
        "${CMAKE_BINARY_DIR}/zlib/lib/${CMAKE_STATIC_LIBRARY_PREFIX}z${CMAKE_STATIC_LIBRARY_SUFFIX}")
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
        -DPATCH_DIR=${CMAKE_CURRENT_LIST_DIR}/patches
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
        # NOTE: USE_OPENCOLORIO must stay ON. color_ocio.cpp is part of
        # libOpenImageIO itself (not a plugin) and includes
        # <OpenColorIO/OpenColorIO.h> and <tsl/robin_map.h> unconditionally,
        # so OCIO, yaml-cpp and Robinmap are hard dependencies of the
        # library on every platform. Their cross-version host-scavenging
        # mismatches are handled by the patch set + OpenImageIO_BUILD_LOCAL_DEPS
        # below instead.
        -DOpenImageIO_BUILD_MISSING_DEPS=required
        # Every image-codec dependency must be sourced from OIIO's own
        # pinned builds, never scavenged from the host. Two reasons:
        #   - Portability/ABI: OIIO would otherwise pull ZLIB/PNG/libjpeg-turbo
        #     from the conda env and Imath/OpenEXR/yaml-cpp from homebrew,
        #     whose dylibs are built for a newer macOS than our deployment
        #     target. OpenImageIO_BUILD_LOCAL_DEPS forces the same
        #     pinned-source set on every platform; local dep builds are static,
        #     land in deps/dist, and the produced dylib ends up depending on
        #     no host image library at all. (This is also what bit us first:
        #     libtiff built against a host libdeflate whose symbols vanished
        #     at the final link because the consuming scope resolved a
        #     different libdeflate than the one tiff was compiled against.)
        #   - Host TIFF is outright incompatible: the macOS wheel env
        #     (.ci/conda_macos.yml) ships no libtiff, so find_package(TIFF) fell
        #     through to the legacy copy bundled in Mono.framework on GitHub
        #     runners; its uint64 is unsigned long while OIIO passes uint64_t*
        #     to TIFFWriteCustomDirectory(), which fails to compile. Listing
        #     TIFF skips find_package(TIFF) entirely and builds OIIO's own
        #     pinned libtiff (src/cmake/build_TIFF.cmake, 4.7.1). OIIO's
        #     build_dependency_with_cmake swallows sub-build failures, so if
        #     that local build ever fails, build_TIFF.cmake's final
        #     find_package(TIFF REQUIRED) would fall back to a host tiff again
        #     - which the next two flags prevent:
        #   - CMAKE_IGNORE_PATH discards any find result under Mono.framework.
        #   - CMAKE_FIND_FRAMEWORK=NEVER removes framework search altogether
        #     (OIIO and its local dep builds need no Apple frameworks).
        # If the local TIFF build ever fails, configure now aborts with an
        # explicit "Could NOT find TIFF" instead of silently compiling the
        # runner's incompatible libtiff.
        #
        # Entries must match OIIO's checked_find_package names: ZLIB, PNG,
        # libjpeg-turbo, Imath, OpenEXR, yaml-cpp. The semicolon-separated
        # list must survive ExternalProject's command-line round-trip, hence
        # the $<SEMICOLON> generator expression, exactly like
        # CMAKE_PREFIX_PATH above.
        -DOpenImageIO_BUILD_LOCAL_DEPS=TIFF$<SEMICOLON>ZLIB$<SEMICOLON>PNG$<SEMICOLON>libjpeg-turbo$<SEMICOLON>Imath$<SEMICOLON>OpenEXR$<SEMICOLON>yaml-cpp
        # OIIO's LOCAL_BUILD_SHARED_LIBS_DEFAULT is ON for local dep builds,
        # which would leave a libz.dylib in deps/dist for the produced dylib
        # to dangle on. Every local dep must ship static.
        # Pin ZLIB to the repo-built zlib 1.3.1 (ext_zlib): the OIIO local
        # deps refind would otherwise pick up the system zlib (1.2.11 on
        # Ubuntu 22.04), which fails the >=1.3.1 version check.
        # Artifact is the static archive on every platform (libz.a /
        # zlibstatic.lib under MSVC) — see _openimageio_zlib_library above
        # for why the shared libz.so/libz.dylib must never be consumed here.
        -DZLIB_LIBRARY=${_openimageio_zlib_library}
        -DZLIB_INCLUDE_DIR=${CMAKE_BINARY_DIR}/zlib/include
        -DZLIB_BUILD_SHARED_LIBS=OFF
        -DCMAKE_IGNORE_PATH=/Library/Frameworks/Mono.framework
        -DCMAKE_FIND_FRAMEWORK=NEVER
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
