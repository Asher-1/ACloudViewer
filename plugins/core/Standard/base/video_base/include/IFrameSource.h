// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// IFrameSource: abstract decode backend for video_base.
//
// Two implementations live behind this interface:
//  - OpenCVFrameSource : cv::VideoCapture (software or best-effort VAAPI /
//                        D3D11 hardware decode), camera input
//  - MpvFrameSource    : libmpv (audio output, hardware decode via mpv's
//                        hwdec=auto, frame-exact hr-seek) — compiled only
//                        when HAS_LIBMPV is defined
//
// The interface is deliberately synchronous ("pull" model): the owning
// VideoFrameReader runs it on a dedicated QThread and turns results into
// frameReady() / frameReadFailed() signals, so VideoPlaybackWidget is
// decoupled from which backend is active.
// ----------------------------------------------------------------------------

#pragma once

#include <string>

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/core.hpp>

class IFrameSource {
public:
    // read() outcome.  NoFrame means "no new frame available yet" (only the
    // mpv backend, which is clock-driven); Eof means the stream ended.
    enum class ReadResult { Ok, NoFrame, Eof };

    virtual ~IFrameSource() = default;

    // Open a video file.  backendHint is an OpenCV backend id (CAP_FFMPEG /
    // CAP_ANY); mpv ignores it.
    virtual bool openVideo(const std::string& path, int backendHint = 0) = 0;

    // Open a camera device (OpenCV only; mpv has no camera input).
    virtual bool openCamera(int deviceIndex, int backendHint = 0) = 0;

    virtual bool isOpened() const = 0;

    // Decode / fetch the next frame (BGR).  frameIndex receives the video
    // frame number when Ok.  On NoFrame / Eof `out` is left untouched.
    virtual ReadResult read(cv::Mat& out, int64_t* frameIndex) = 0;

    virtual bool seekToFrame(int frameIndex) = 0;

    virtual int64_t frameCount() const = 0;
    virtual double fps() const = 0;
    virtual int width() const = 0;
    virtual int height() const = 0;

    // Stop / release the underlying source.  Video-file semantics keep the
    // file open for resume; camera semantics release the device.
    virtual void release() = 0;

    // Optional playback controls.  Default implementations are no-ops and
    // report "not supported"; OpenCV playback pacing is done by the widget
    // timers, while mpv needs the properties for A/V-synced playback.
    virtual bool setPaused(bool paused) {
        (void)paused;
        return false;
    }
    virtual bool setSpeed(double speed) {
        (void)speed;
        return false;
    }
};

#endif  // HAS_OPENCV_FACE_CAPTURE
