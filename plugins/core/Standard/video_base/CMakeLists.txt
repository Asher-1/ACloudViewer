# ----------------------------------------------------------------------------
# -                        CloudViewer: www.cloudViewer.org                  -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.cloudViewer.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
# video_base: shared camera / video-file playback library for GGML-based
# inference plugins (qFreeSplatter, qFaceDetect, future plugins).
#
# Provides VideoPlaybackWidget (playback UI + control + frame pipeline) and
# VideoFrameReader (background-thread decode).  Links OpenCV videoio when
# available; HAS_OPENCV_FACE_CAPTURE is propagated PUBLIC so consumers can
# conditionally compile capture code.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION ${CLOUDVIEWER_SUBPROJECT_CMAKE_MINIMUM_VERSION})

project( video_base )

# Optional libmpv backend: audio output + hardware decode (hwdec=auto) +
# frame-exact seeking, at the cost of a few MB of runtime.  Disable for
# size-sensitive builds — the OpenCV backend (with best-effort VAAPI/D3D11
# hardware decode) is used automatically.
option(PLUGIN_VIDEO_LIBMPV "Build video_base with the libmpv playback backend (audio + hardware decode)" ON)

add_library( video_base STATIC )

target_sources( video_base
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/IFrameSource.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/OpenCVFrameSource.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/MpvFrameSource.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/VideoFrameReader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/VideoPlaybackWidget.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/OpenCVFrameSource.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/MpvFrameSource.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/VideoFrameReader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/src/VideoPlaybackWidget.cpp
)

target_include_directories( video_base
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties( video_base PROPERTIES AUTOMOC ON )

# ecvClickableImageLabel / cvFileDialog / ecvPS live in CVPluginAPI.
target_link_libraries( video_base
    PUBLIC
        CVPluginAPI
        Qt::Widgets
        Qt::Concurrent
)

# OpenCV videoio is optional: without it the module degrades gracefully
# (VideoPlaybackWidget::isAvailable() == false).
if( TARGET 3rdparty_opencv )
    target_link_libraries( video_base PUBLIC 3rdparty_opencv )
    target_compile_definitions( video_base PUBLIC HAS_OPENCV_FACE_CAPTURE )
    message( STATUS "video_base: OpenCV video capture enabled (HAS_OPENCV_FACE_CAPTURE)" )
else()
    message( STATUS "video_base: OpenCV video capture disabled (no 3rdparty_opencv)" )
endif()

# libmpv backend: detected via pkg-config on platforms where libmpv is
# available.  Linux uses the system libmpv (apt libmpv-dev provides .pc);
# macOS uses the conda-forge mpv package (conda_*.yml).  Windows falls
# back to OpenCV (conda-forge does not ship mpv for win-64).
# When libmpv is not available the build silently falls back to the OpenCV
# backend — CI never fails.
if( PLUGIN_VIDEO_LIBMPV )
    find_package(PkgConfig QUIET)
    if( PKG_CONFIG_FOUND )
        pkg_check_modules(3rdparty_mpv QUIET IMPORTED_TARGET mpv)
    endif()
    if( TARGET PkgConfig::3rdparty_mpv )
        target_link_libraries( video_base PUBLIC PkgConfig::3rdparty_mpv )
        target_compile_definitions( video_base PUBLIC HAS_LIBMPV )
        message( STATUS "video_base: libmpv backend enabled (audio + hardware decode + frame-exact seek)" )
    else()
        message( STATUS "video_base: libmpv not found — OpenCV-only decode" )
    endif()
endif()

if( MSVC )
    target_compile_definitions( video_base PUBLIC _USE_MATH_DEFINES )
endif()
