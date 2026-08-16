// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// OpenCVFrameSource: IFrameSource implementation backed by cv::VideoCapture.
//
// Extracted from the original VideoFrameReader; adds best-effort hardware
// decode (VAAPI on Linux / D3D11 on Windows, OpenCV >= 4.5.2) and exact-seek
// frame alignment for VFR / keyframe-boundary correctness on Windows.
// ----------------------------------------------------------------------------

#pragma once

#include "IFrameSource.h"

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/videoio.hpp>

class OpenCVFrameSource : public IFrameSource {
public:
    bool openVideo(const std::string& path, int backendHint = 0) override;
    bool openCamera(int deviceIndex, int backendHint = 0) override;

    bool isOpened() const override;
    ReadResult read(cv::Mat& out, int64_t* frameIndex) override;
    bool seekToFrame(int frameIndex) override;

    int64_t frameCount() const override;
    double fps() const override;
    int width() const override;
    int height() const override;

    void release() override;

    // Shared by VideoPlaybackWidget for the seek-preview capture: opens
    // `cap` with the same best-effort HW-accelerated path as openVideo().
    static bool openVideoWithHw(cv::VideoCapture& cap, const std::string& path,
                                int backend);

private:
    cv::VideoCapture m_cap;

    // Target frame of the most recent seekToFrame().  read() drains frames
    // after the seek until this index is reached (VFR videos and
    // keyframe-boundary seeks can otherwise land a few frames off, which
    // breaks exact-position scrubbing on Windows).  -1 = no pending seek.
    // Accessed only on the reader thread (slots are queued, serialized).
    int m_seekPendingFrame = -1;
};

#endif  // HAS_OPENCV_FACE_CAPTURE
