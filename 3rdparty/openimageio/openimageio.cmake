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

# OIIO is delivered statically on every platform (ceres/lapack policy), so
# the archives embed into libCloudViewer/pybind and no shared payload exists
# to bundle: packaging needs no OIIO deployment, collection, or rpath handling
# anywhere. OIIO's own local-dep system would otherwise default to shared
# libraries under MSVC, which would leave a static OIIO depending on a pile of
# deps DLLs that no packaging search path covers, so LOCAL_BUILD_SHARED_LIBS_DEFAULT
# is forced off and the static closure is discovered after the build through a
# linker response file (see generate_static_closure.cmake). The PIC switch is
# required for Linux, where the static closure links into the pybind module;
# patch 0002 propagates it to every local dep sub-build.
if(WIN32)
    set(_openimageio_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/OpenImageIO.lib")
    set(_openimageio_util_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/OpenImageIO_Util.lib")
else()
    set(_openimageio_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}OpenImageIO${CMAKE_STATIC_LIBRARY_SUFFIX}")
    set(_openimageio_util_link
        "<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}OpenImageIO_Util${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

# ext_zlib artifact consumed via ZLIB_LIBRARY below, pinned to the STATIC
# archive on every platform. zlib always builds the static `zlibstatic`
# target; non-MSVC renames it to `z` (libz.a) alongside the shared
# `libz.so`/`libz.dylib`, while MSVC builds with BUILD_SHARED_LIBS=OFF and
# keeps `zlibstatic.lib`. The static pin is required, not stylistic: a shared
# zlib would embed a build-tree install name (macOS records the full dylib
# path in LC_LOAD_DYLIB, e.g. @rpath/libz.1.dylib pointing outside deps/dist)
# into the static closure, reintroducing exactly the cross-payload runtime
# dependency the all-static delivery exists to eliminate. Embedding zlib
# keeps the OIIO runtime closure to its own two archives — the same
# static-only rule the local dep builds enforce (patches/build_ZLIB.cmake
# deletes OIIO's own shared libz for exactly this reason). Never hardcode a
# Linux-only suffix here either: the path is embedded into OIIO's sub-build
# as a hard file prerequisite, so a missing artifact fails it with
# "No rule to make target" (make) / LNK1181 (MSVC).
#
# The pin must be the STATIC archive on every platform. FindZLIB skips its
# release/debug search and select_library_configurations() entirely when
# ZLIB_LIBRARY is preset, so on a fresh configure ZLIB_LIBRARY_RELEASE is
# never populated and ZLIB::ZLIB imports exactly this file. Pinning the
# shared dylib would therefore embed @rpath/libz.1.dylib into libOpenImageIO:
# a dylib outside the static closure that no consumer ships, so the import
# fails at load time - pinning the archive keeps every entry of the rsp a
# real static dependency.
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
        -DBUILD_SHARED_LIBS=OFF
        # OIIO's local-dep system defaults to shared libraries under MSVC
        # (upstream never got them working static there); force the same
        # static-deps policy everywhere so the static archive closes over
        # deps/dist on every platform. See generate_static_closure.cmake.
        -DLOCAL_BUILD_SHARED_LIBS_DEFAULT=OFF
        # The static closure links into the pybind module (a shared object);
        # Linux rejects non-PIC archives in -fPIC links. patch 0001 propagates
        # this to every local dep sub-build.
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
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
        # libjpeg-turbo, Imath, OpenEXR, yaml-cpp, minizip-ng, expat. The
        # semicolon-separated list must survive ExternalProject's command-
        # line round-trip, hence the $<SEMICOLON> generator expression,
        # exactly like CMAKE_PREFIX_PATH above.
        # minizip-ng must be in this list, not merely a missing-deps fallback:
        # a host carrying minizip-ng >= 4.0.10 would otherwise be found by
        # find_package and silently scavenged into the closure instead of the
        # pinned build (whose 0001 patch renames mz_zip_writer_add_file so it
        # can never collide with the miniz copy embedded in libassimp.a).
        # Forcing the local
        # build also guarantees the rename flag is always compiled in.
        # expat is the same story (ubuntu-noble CI): noble ships expat 2.6.x,
        # which satisfies OIIO's VERSION_MIN, so OCIO links the host SHARED
        # expat, no libexpat.a lands in deps/dist, and the static closure
        # cannot resolve OCIO's XML_* references. Pinning the local build
        # keeps expat inside deps/dist on every platform. yaml-cpp is already
        # on this list upstream, so the same short-circuit protects the
        # windows wheel from host-yaml-cpp pollution (see the OCIO pins in
        # build_OpenColorIO.cmake inside patch 0001).
        -DOpenImageIO_BUILD_LOCAL_DEPS=TIFF$<SEMICOLON>ZLIB$<SEMICOLON>PNG$<SEMICOLON>libjpeg-turbo$<SEMICOLON>Imath$<SEMICOLON>OpenEXR$<SEMICOLON>yaml-cpp$<SEMICOLON>minizip-ng$<SEMICOLON>expat
        # OIIO's LOCAL_BUILD_SHARED_LIBS_DEFAULT is ON for local dep builds,
        # which would leave a libz.dylib in deps/dist for the produced dylib
        # to dangle on. Every local dep must ship static.
        # Pin ZLIB to the repo-built zlib 1.3.1 (ext_zlib): the OIIO local
        # deps refind would otherwise pick up the system zlib (1.2.11 on
        # Ubuntu 22.04), which fails the >=1.3.1 version check.
        # Artifact is the static archive on every platform (libz.a /
        # zlibstatic.lib under MSVC) — see _openimageio_zlib_library above
        # for why the shared libz.so/libz.dylib/libz.1.dylib must never be
        # consumed here.
        -DZLIB_LIBRARY=${_openimageio_zlib_library}
        -DZLIB_INCLUDE_DIR=${CMAKE_BINARY_DIR}/zlib/include
        -DZLIB_BUILD_SHARED_LIBS=OFF
        -DCMAKE_IGNORE_PATH=/Library/Frameworks/Mono.framework
        -DCMAKE_FIND_FRAMEWORK=NEVER
    DEPENDS ext_zlib)

# The patch step is stamp-gated on the extracted source tree alone, and
# UPDATE_COMMAND is empty: editing a patch file in the repo used to leave
# every existing build tree building with the previous patch set (observed
# as the mz_zip_writer_add_file duplicate-symbol link failure after the
# rename was added to 0001). Tie the patch step to the patch files so a
# patch edit re-runs it; on a tree that no longer matches, the apply script
# aborts with wipe-and-re-extract instructions instead of silently keeping
# the stale binaries.
file(GLOB _openimageio_patch_files "${CMAKE_CURRENT_LIST_DIR}/patches/*.patch")
ExternalProject_Add_StepDependencies(ext_openimageio patch ${_openimageio_patch_files})
# NOTE for patch authors: a changed patch set invalidates more than the
# source tree. Wipe the whole ext_openimageio prefix (source AND build
# directories) before rebuilding -- stale dep caches and clones (e.g. an
# old Imath_BUILD_VERSION or a shallow clone pinned to a bumped tag)
# otherwise conflict with the new patch set at configure time.

# Discover the static closure after the install finished. The step reruns on
# every OIIO build so the rsp never goes stale.
#
# Apple/GNU linkers expand @rsp from their own response files, so those
# platforms consume the rsp directly as a link item. MSVC link.exe cannot
# expand a nested @rsp (MSBuild/Ninja wrap every link line in their own
# response file, and link.exe treats the inner @path as a plain library
# input, failing with "LNK1104: cannot open file '@...rsp.lib'"). On Windows
# the closure is therefore physically merged into ONE archive by lib.exe
# (CMAKE_AR) inside this step, and the consumer links that single known-name
# archive. MSVC resolves static-archive members with repeated scans, so the
# merged archive needs neither the doubled entries nor any ordering the
# two-pass rsp relies on.
if(MSVC)
    if(NOT CMAKE_AR)
        message(FATAL_ERROR
            "OIIO static closure: MSVC builds merge the closure with "
            "CMAKE_AR (lib.exe), but CMAKE_AR is unset. Configure with a "
            "complete MSVC toolchain before enabling BUILD_RECONSTRUCTION.")
    endif()
    set(_openimageio_closure_rsp
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/oiio_static_closure.rsp)
    set(_openimageio_closure_merged
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/oiio_static_closure.lib)
    set(_openimageio_closure_byproducts
        BYPRODUCTS ${_openimageio_closure_rsp} ${_openimageio_closure_merged})
    set(_openimageio_closure_extra_args -DMERGE_LIB_TOOL=${CMAKE_AR})
else()
    set(_openimageio_closure_rsp
        <INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/oiio_static_closure.rsp)
    set(_openimageio_closure_byproducts
        BYPRODUCTS ${_openimageio_closure_rsp})
    set(_openimageio_closure_extra_args "")
endif()

ExternalProject_Add_Step(ext_openimageio generate_static_closure
    COMMAND ${CMAKE_COMMAND}
        -DLIB_PREFIX=${CMAKE_STATIC_LIBRARY_PREFIX}
        -DLIB_SUFFIX=${CMAKE_STATIC_LIBRARY_SUFFIX}
        -DOIIO_INSTALL_LIB=<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}
        -DOIIO_BINARY_DIR=<BINARY_DIR>
        -DEXTRA_LIBS=${_openimageio_zlib_library}
        -DOUT_RSP=<INSTALL_DIR>/${CloudViewer_INSTALL_LIB_DIR}/oiio_static_closure.rsp
        ${_openimageio_closure_extra_args}
        -P ${CMAKE_CURRENT_LIST_DIR}/generate_static_closure.cmake
    DEPENDEES install
    ${_openimageio_closure_byproducts}
    )

ExternalProject_Get_Property(ext_openimageio INSTALL_DIR)

# Consumed through the shared import_3rdparty_library / import_shared_3rdparty_library
# interface in find_dependencies.cmake, exactly like ext_ceres: the triple of
# variables below is the only contract, so the platform-dependent link form
# (macOS static archives vs Linux/Windows shared libraries) is decided by the
# same helper every other ExternalProject goes through.
set(OPENIMAGEIO_INCLUDE_DIRS "${INSTALL_DIR}/include")
set(OPENIMAGEIO_LIB_DIR "${INSTALL_DIR}/${CloudViewer_INSTALL_LIB_DIR}")
set(EXT_OPENIMAGEIO_LIBRARIES OpenImageIO OpenImageIO_Util)
# Static closure discovered after the build (see generate_static_closure.cmake):
# GNU/Apple linkers consume the rsp file; MSVC links the lib.exe-merged
# single archive (see the step comment above for why nested @rsp fails).
set(OPENIMAGEIO_RSP_FILE "${OPENIMAGEIO_LIB_DIR}/oiio_static_closure.rsp")
set(OPENIMAGEIO_MERGED_LIB "${OPENIMAGEIO_LIB_DIR}/oiio_static_closure.lib")

message(STATUS
    "Reconstruction image backend: OpenImageIO ${OPENIMAGEIO_VERSION} from 3rdparty source (OpenCV disabled)")
