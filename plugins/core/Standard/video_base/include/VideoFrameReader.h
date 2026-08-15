// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// VideoFrameReader: owns a cv::VideoCapture on a dedicated QThread so that
// OpenCV's MSMF/DirectShow backend (Windows) does not block the Qt main
// thread during read() calls.  Communicates frames as BGR cv::Mat via an
// emitted signal for direct use by downstream consumers (face detection,
// model inference, display).
//
// Part of the video_base module shared by qFreeSplatter / qFaceDetect and
// any future plugin that needs camera / video-file input.
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>

#include <atomic>
#include <string>

#ifdef HAS_OPENCV_FACE_CAPTURE
#include <opencv2/videoio.hpp>

class VideoFrameReader : public QObject {
    Q_OBJECT
public:
    explicit VideoFrameReader(QObject* parent = nullptr);
    ~VideoFrameReader() override;

    // Open a video file; falls back to CAP_ANY when the requested backend
    // cannot handle the container/codec.
    bool openVideo(const std::string& path, int backend = cv::CAP_ANY);

    // Open a camera device; requests 640x480 to keep decode cost bounded.
    bool openCamera(int deviceIndex, int backend = cv::CAP_ANY);

    bool isOpened() const;

    int64_t getFrameCount() const;
    double getFps() const;
    int getFrameWidth() const;
    int getFrameHeight() const;
    int currentFrameNum() const;

public slots:
    // Decode one frame and emit frameReady() / frameReadFailed().
    // Reentrancy guard: on Windows the MSMF/DirectShow backend can take
    // 50-200 ms per read().  If the UI timer keeps firing faster than
    // that, queued readFrame calls would pile up and the video would
    // race ahead / lag behind indefinitely.  Drop redundant reads.
    void readFrame();

    // Seek to a frame index (queued calls are serialized on the reader
    // thread, so no mutex is needed here).
    void seekToFrame(int frameIndex);

    // Release the underlying capture (invoked from the GUI thread when
    // stopping the reader thread).
    Q_INVOKABLE void release();

signals:
    // Delivers the raw BGR frame.  Downstream consumers (cvMatToQImage /
    // detection) perform the single BGR->RGB conversion; converting here
    // would double-swap the channels and corrupt colors.
    void frameReady(const cv::Mat& rgbFrame, int frameIndex);

    // Emitted when a read fails or returns an empty frame (e.g. video
    // end-of-file).
    void frameReadFailed();

private:
    cv::VideoCapture m_cap;
    std::atomic<bool> m_reading{false};
};

#else  // !HAS_OPENCV_FACE_CAPTURE

// No OpenCV build: video capture is unavailable.  Consumers must guard all
// usage with HAS_OPENCV_FACE_CAPTURE (VideoPlaybackWidget::isAvailable()).
class VideoFrameReader : public QObject {
    Q_OBJECT
public:
    explicit VideoFrameReader(QObject* parent = nullptr) : QObject(parent) {}
};

#endif  // HAS_OPENCV_FACE_CAPTURE
