# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
# spirula_base: shared Spirula Studio support for plugins (qSpirula,
# qJSonRPCPlugin, future consumers) — executable location + launch, and the
# light 3DGS-PLY / mesh importer that lands training outputs in the DB tree.
#
# Deliberately built UNCONDITIONALLY (no BUILD_SPIRULA guard): it is a pure
# Qt/CC static library with zero spirula dependencies, and both consumers rely
# on its target always existing — qSpirula for its dialog, qJSonRPCPlugin for
# the spirula.* RPC methods (they self-report a clear error at runtime when
# the executable is absent). Splitting it out of qSpirula is required because
# plugins cannot link each other — qJSonRPCPlugin is a second, real consumer
# (same reason video_base exists for qFreeSplatter/qFaceDetect).
#
# Process-boundary contract with the standalone `spirula` executable built by
# 3rdparty/spirula/spirula.cmake (BUILD_SPIRULA): no source patches, sync =
# SPIRULA_TAG bump + smoke tests.  Compile-time fallback path is propagated
# as the QSPIRULA_BUILD_PATH definition.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project( spirula_base )

add_library( spirula_base STATIC )

target_sources( spirula_base
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/src/SpirulaLocator.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/SpirulaLauncher.cpp
        ${CMAKE_CURRENT_LIST_DIR}/src/SplatImporter.cpp
)

target_include_directories( spirula_base
    PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

# SplatImporter builds ccPointCloud entities (CVCoreLib) and loads meshes via
# the regular I/O stack (CV_IO_LIB); Qt Core covers its file helpers.
target_link_libraries( spirula_base
    PUBLIC
        CVCoreLib
        CV_IO_LIB
        Qt${Qt_VERSION_MAJOR}::Core
)
