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

add_library( video_base STATIC )

target_sources( video_base
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/VideoFrameReader.h
        ${CMAKE_CURRENT_SOURCE_DIR}/include/VideoPlaybackWidget.h
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

if( MSVC )
    target_compile_definitions( video_base PUBLIC _USE_MATH_DEFINES )
endif()
