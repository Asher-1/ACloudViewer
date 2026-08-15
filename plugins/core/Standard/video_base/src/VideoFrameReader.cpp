// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoFrameReader.h"

#ifdef HAS_OPENCV_FACE_CAPTURE

VideoFrameReader::VideoFrameReader(QObject* parent) : QObject(parent) {}

VideoFrameReader::~VideoFrameReader() {
    release();
}

bool VideoFrameReader::openVideo(const std::string& path, int backend) {
    release();
    return m_cap.open(path, backend) || m_cap.open(path, cv::CAP_ANY);
}

bool VideoFrameReader::openCamera(int deviceIndex, int backend) {
    release();
    m_cap.open(deviceIndex, backend);
    if (m_cap.isOpened()) {
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    }
    return m_cap.isOpened();
}

bool VideoFrameReader::isOpened() const {
    return m_cap.isOpened();
}

int64_t VideoFrameReader::getFrameCount() const {
    return m_cap.isOpened() ? static_cast<int64_t>(
                                      m_cap.get(cv::CAP_PROP_FRAME_COUNT))
                            : 0;
}

double VideoFrameReader::getFps() const {
    return m_cap.isOpened() ? m_cap.get(cv::CAP_PROP_FPS) : 0.0;
}

int VideoFrameReader::getFrameWidth() const {
    return m_cap.isOpened()
                   ? static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH))
                   : 0;
}

int VideoFrameReader::getFrameHeight() const {
    return m_cap.isOpened()
                   ? static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                   : 0;
}

int VideoFrameReader::currentFrameNum() const {
    return m_cap.isOpened() ? static_cast<int>(m_cap.get(cv::CAP_PROP_POS_FRAMES))
                            : 0;
}

void VideoFrameReader::readFrame() {
    // Reentrancy guard: on Windows the MSMF/DirectShow backend can take
    // 50-200 ms per read().  If the UI timer keeps firing faster than
    // that, queued readFrame calls would pile up and the video would
    // race ahead / lag behind indefinitely.  Drop redundant reads.
    if (m_reading.exchange(true)) return;
    if (!m_cap.isOpened()) {
        m_reading = false;
        return;
    }
    cv::Mat frame;
    const bool ok = m_cap.read(frame) && !frame.empty();
    m_reading = false;
    if (!ok) {
        emit frameReadFailed();
        return;
    }
    // Emit the raw OpenCV frame (BGR).  Downstream consumers
    // (cvMatToQImage / detection) perform the single BGR->RGB conversion;
    // converting here would double-swap the channels and corrupt colors.
    emit frameReady(frame, currentFrameNum());
}

void VideoFrameReader::seekToFrame(int frameIndex) {
    if (m_cap.isOpened()) {
        m_cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    }
}

void VideoFrameReader::release() {
    if (m_cap.isOpened()) m_cap.release();
}

#endif  // HAS_OPENCV_FACE_CAPTURE
