# Spirula Studio — a standalone 3D Gaussian Splatting trainer (GPLv3), built
# as an ExternalProject *executable* and consumed by the qSpirula plugin
# through a process boundary (QProcess + result files).
#
# Deliberately zero patches: upstream synchronization is "bump SPIRULA_TAG,
# rebuild, run the smoke tests". The CMake args mirror the upstream release
# CI (.github/workflows/ci.yml) verbatim:
#
#     build_develop.bash / build_develop.bat -DSS_BACKEND=vulkan -DSS_ENABLE_PATENTED=ON
#
# Every other SS_* switch keeps its upstream default (SS_BUILD_GUI=ON,
# SS_BUILD_SAM=ON, SS_BUILD_SFM=ON, SS_FONT_CJK=fetch, ...), so release
# parity is structural rather than a maintained switch mapping.
#
# CUDA is intentionally not a distribution config: the CUDA backend dynamically
# links cudart and embeds SASS for the build host's GPU (upstream default
# SS_CUDA_EMBED_PTX=OFF), so the result cannot be redistributed across driver/
# GPU combinations. Build a local variant with -DSPIRULA_BACKEND=cuda; the
# qSpirula plugin prefers a spirula-cuda binary next to the app at runtime
# when an NVIDIA driver answers.
#
# Hard upstream constraint honored here: SS_ENABLE_PATENTED=ON requires
# SS_BUILD_SAM=ON (the Vulkan Video decoder is built on the src/nn/ runtime).
# The Vulkan backend already defaults SAM to ON; the CUDA backend defaults it
# OFF, which is the one place we must deviate from "inherit upstream defaults"
# to keep the same feature set across variants.

if(NOT BUILD_SPIRULA)
    return()
endif()

include(ExternalProject)

set(SPIRULA_REPO_URL "https://github.com/harry7557558/spirula-studio"
        CACHE STRING "Spirula Studio repository the tag archive is fetched from")
set(SPIRULA_TAG "v2026.10.6" CACHE STRING "Spirula Studio tag to build (sync = bump this)")
set(SPIRULA_SLANG_VERSION "2026.12.0.1" CACHE STRING
        "slang release version matching SPIRULA_TAG's pinned SS_SLANG_VERSION (bump together with the tag; only the Linux glibc-2.28 fallback below uses it)")
set(SPIRULA_BACKEND "vulkan" CACHE STRING "Spirula compute backend: vulkan (release) | cuda (local only)")
set(SPIRULA_SLANGC "" CACHE FILEPATH
        "Pinned slangc for offline builds (forwarded as -DSS_SLANGC; empty = upstream auto-fetch of the pinned release)")

if(NOT SPIRULA_BACKEND STREQUAL "vulkan" AND NOT SPIRULA_BACKEND STREQUAL "cuda")
    message(FATAL_ERROR "SPIRULA_BACKEND must be 'vulkan' or 'cuda', got '${SPIRULA_BACKEND}'")
endif()

set(SPIRULA_EXE_NAME "spirula${CMAKE_EXECUTABLE_SUFFIX}")

# Cold-cache convenience: pick up a slang toolchain previously fetched into
# the download cache (e.g. the glibc-2.28 variant required on Ubuntu 20.04,
# whose glibc 2.31 cannot run the default asset which needs >= 2.33). Without
# this, a fresh checkout that forgets -DSPIRULA_SLANGC would download the
# default asset and fail at slangc version probe on older hosts. A cached
# slangc with a mismatched version still fails the upstream version probe
# explicitly, so a stale cache cannot be silently used.
if(NOT SPIRULA_SLANGC)
    file(GLOB _spirula_cached_slangc
         "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/spirula/slang-*/bin/slangc")
    if(_spirula_cached_slangc)
        list(GET _spirula_cached_slangc 0 _spirula_cached_pick)
        set(SPIRULA_SLANGC "${_spirula_cached_pick}" CACHE FILEPATH
                "slangc picked up from the download cache" FORCE)
        message(STATUS "SPIRULA_SLANGC picked from cache: ${SPIRULA_SLANGC}")
    endif()
    unset(_spirula_cached_slangc)
    unset(_spirula_cached_pick)
endif()

# Host glibc too old for the default linux slang asset (it needs >= 2.33;
# Ubuntu 20.04 ships 2.31, and the CI matrix builds inside a cpu-focal
# container)? Fetch the same version's glibc-2.28 variant into the download
# cache so fresh checkouts work without manual slang setup. Runs once per
# configure; the extracted toolchain is then picked up by the probe above on
# every later re-configure. macOS/Windows use different asset families and
# are not affected.
if(NOT SPIRULA_SLANGC AND UNIX AND NOT APPLE)
    set(_spirula_slang_dir "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/spirula/slang-${SPIRULA_SLANG_VERSION}-linux-x86_64-glibc-2.28")
    if(NOT EXISTS "${_spirula_slang_dir}/bin/slangc")
        execute_process(COMMAND ldd --version OUTPUT_VARIABLE _spirula_ldd_out ERROR_QUIET)
        # Matches "ldd (GNU libc) 2.31" and "ldd (Ubuntu GLIBC 2.31-x) 2.31":
        # take the pure version that follows the closing parenthesis.
        string(REGEX MATCH "ldd \\([^)]*\\) 2\\.([0-9]+)" _spirula_glibc "${_spirula_ldd_out}")
        if(_spirula_glibc AND CMAKE_MATCH_1 LESS 33)
            set(_spirula_slang_url "https://github.com/shader-slang/slang/releases/download/v${SPIRULA_SLANG_VERSION}/slang-${SPIRULA_SLANG_VERSION}-linux-x86_64-glibc-2.28.tar.gz")
            set(_spirula_slang_tgz "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/spirula/slang-${SPIRULA_SLANG_VERSION}-linux-x86_64-glibc-2.28.tar.gz")
            if(NOT EXISTS "${_spirula_slang_tgz}")
                message(STATUS "SPIRULA: host glibc < 2.33 — downloading the glibc-2.28 slang ${SPIRULA_SLANG_VERSION} build")
                file(DOWNLOAD "${_spirula_slang_url}" "${_spirula_slang_tgz}"
                        SHOW_PROGRESS STATUS _spirula_slang_dl)
                list(GET _spirula_slang_dl 0 _spirula_slang_rc)
                if(NOT _spirula_slang_rc EQUAL 0)
                    message(FATAL_ERROR
                            "Failed to download the glibc-2.28 slang build: ${_spirula_slang_dl}. "
                            "Install a slangc ${SPIRULA_SLANG_VERSION} this host can run and pass -DSPIRULA_SLANGC.")
                endif()
            endif()
            # The tarball has no top-level directory.
            file(ARCHIVE_EXTRACT INPUT "${_spirula_slang_tgz}" DESTINATION "${_spirula_slang_dir}")
            set(SPIRULA_SLANGC "${_spirula_slang_dir}/bin/slangc" CACHE FILEPATH
                    "slangc fetched for this host's glibc" FORCE)
            message(STATUS "SPIRULA_SLANGC fetched for old-glibc host: ${SPIRULA_SLANGC}")
        endif()
        unset(_spirula_glibc)
        unset(_spirula_ldd_out)
    endif()
    unset(_spirula_slang_dir)
endif()

set(SPIRULA_EXTRA_CMAKE_ARGS)
if(SPIRULA_SLANGC)
    list(APPEND SPIRULA_EXTRA_CMAKE_ARGS "-DSS_SLANGC=${SPIRULA_SLANGC}")
endif()
if(SPIRULA_BACKEND STREQUAL "cuda")
    # Upstream defaults SS_BUILD_SAM=OFF on the CUDA backend, but
    # SS_ENABLE_PATENTED=ON hard-requires it (SsOptions.cmake FATAL_ERROR).
    list(APPEND SPIRULA_EXTRA_CMAKE_ARGS "-DSS_BUILD_SAM=ON")
endif()

ExternalProject_Add(ext_spirula
    # Build tree and install tree under build_app/spirula/ — CMake's default
    # INSTALL_DIR is the PREFIX itself, so the executable lands flat in
    # build_app/spirula/bin/, the same layout as build_app/fmt and
    # build_app/libpng. 3rdparty_downloads holds only the downloaded tarball.
    PREFIX spirula
    URL "${SPIRULA_REPO_URL}/archive/refs/tags/${SPIRULA_TAG}.tar.gz"
    DOWNLOAD_DIR "${CLOUDVIEWER_THIRD_PARTY_DOWNLOAD_DIR}/spirula"
    DOWNLOAD_NAME "spirula-${SPIRULA_TAG}.tar.gz"
    # Release-parity args: identical to the upstream CI invocation. Only these
    # two -D switches plus the install prefix are explicit on purpose.
    CMAKE_ARGS ${ExternalProject_CMAKE_ARGS_hidden}
        -DSS_BACKEND=${SPIRULA_BACKEND}
        -DSS_ENABLE_PATENTED=ON
        -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
        ${SPIRULA_EXTRA_CMAKE_ARGS}
    # Build ONLY the shipping executable: upstream's own CI owns the parity/
    # unit test targets, and they occasionally lag the shipping tree (e.g.
    # v2026.10.6's heif_test fails to link against a refactored VideoPipeline).
    # This is a scope reduction, not a source patch.
    BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --target spirula --config $<CONFIG>
    BUILD_BYPRODUCTS "<INSTALL_DIR>/bin/${SPIRULA_EXE_NAME}"
    # The tag archive has no .git, so upstream's source lints would scope to
    # the enclosing ACloudViewer checkout; give the source its own git
    # identity (see spirula_git_init.cmake — environment repair, not a patch).
    PATCH_COMMAND ${CMAKE_COMMAND}
        -DSS_SOURCE_DIR=<SOURCE_DIR>
        -P ${CMAKE_CURRENT_LIST_DIR}/spirula_git_init.cmake
    # Upstream ships no install rules; the executable is the whole artifact.
    INSTALL_COMMAND
        ${CMAKE_COMMAND} -E copy_if_different
        <BINARY_DIR>/spirula${CMAKE_EXECUTABLE_SUFFIX}
        <INSTALL_DIR>/bin/${SPIRULA_EXE_NAME}
    VERBATIM)

ExternalProject_Get_Property(ext_spirula INSTALL_DIR)
set(SPIRULA_EXE "${INSTALL_DIR}/bin/${SPIRULA_EXE_NAME}")
set(SPIRULA_EXECUTABLE "${SPIRULA_EXE}" CACHE INTERNAL
        "Path of the built Spirula Studio executable (consumed by qSpirula)")

# Shipping location: next to the main application so SpirulaLocator's first
# probe (applicationDirPath) hits it on every platform — Linux/Windows install
# into the app folder (bin/ACloudViewer/), macOS into the bundle's
# Contents/MacOS. The plugin-scan directories (bin/plugins, Contents/cvPlugins)
# stay purely Qt-plugin modules; spirula is an auxiliary executable, not a
# scanned module. The executable is self-contained (ldd: libc/libgomp/
# libglvnd/libvulkan-loader only — the same Vulkan loader dependency the main
# app already carries), so nothing else needs to be bundled.
if(APPLE)
    # ACloudViewer_MAC_BASE_DIR is defined in app/Mac/CMakeLists.txt at the
    # build-tree scope; resolve it in the install script instead of duplicating
    # the bundle-path formula here.
    install(CODE
        "execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
            \"${SPIRULA_EXE}\"
            \"${ACLOUDVIEWER_MAC_BASE_DIR}/Contents/MacOS/${SPIRULA_EXE_NAME}\")")
else()
    install(PROGRAMS "${SPIRULA_EXE}" DESTINATION ${ACloudViewer_DEST_FOLDER})
endif()
