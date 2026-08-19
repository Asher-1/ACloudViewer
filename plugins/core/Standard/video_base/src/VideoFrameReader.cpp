// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VideoFrameReader.h"

#ifdef HAS_OPENCV_FACE_CAPTURE

VideoFrameReader::VideoFrameReader(QObject* parent) : QObject(parent) {
    resetSource();
}

void VideoFrameReader::resetSource() {
    if (m_consumerDriven) {
        m_source = std::make_unique<OpenCVFrameSource>();
        return;
    }
#ifdef HAS_LIBMPV
    if (MpvFrameSource::available()) {
        m_source = std::make_unique<MpvFrameSource>();
    } else {
        m_source = std::make_unique<OpenCVFrameSource>();
    }
#else
    m_source = std::make_unique<OpenCVFrameSource>();
#endif
}

VideoFrameReader::~VideoFrameReader() { release(); }

void VideoFrameReader::setConsumerDriven(bool enabled) {
    if (m_consumerDriven == enabled && m_source) return;
    release();
    m_consumerDriven = enabled;
    resetSource();
}

bool VideoFrameReader::openVideo(const std::string& path, int backend) {
    release();
    resetSource();
    if (m_source->openVideo(path, backend)) return true;
#ifdef HAS_LIBMPV
    // The mpv backend could not open the file (unsupported container /
    // missing demuxer) — retry with the OpenCV path for this file.
    if (dynamic_cast<MpvFrameSource*>(m_source.get())) {
        m_source = std::make_unique<OpenCVFrameSource>();
        return m_source->openVideo(path, backend);
    }
#endif
    return false;
}

bool VideoFrameReader::openCamera(int deviceIndex, int backend) {
    release();
#ifdef HAS_LIBMPV
    // Cameras are always handled by OpenCV (mpv has no camera input).
    if (dynamic_cast<MpvFrameSource*>(m_source.get())) {
        m_source = std::make_unique<OpenCVFrameSource>();
    }
#endif
    return m_source->openCamera(deviceIndex, backend);
}

bool VideoFrameReader::isOpened() const {
    return m_source ? m_source->isOpened() : false;
}

int64_t VideoFrameReader::getFrameCount() const {
    return m_source ? m_source->frameCount() : 0;
}

double VideoFrameReader::getFps() const {
    return m_source ? m_source->fps() : 0.0;
}

int VideoFrameReader::getFrameWidth() const {
    return m_source ? m_source->width() : 0;
}

int VideoFrameReader::getFrameHeight() const {
    return m_source ? m_source->height() : 0;
}

int VideoFrameReader::currentFrameNum() const { return m_lastFrameIndex; }

void VideoFrameReader::readFrame() {
    // Reentrancy guard: on Windows the MSMF/DirectShow backend can take
    // 50-200 ms per read().  If the UI timer keeps firing faster than
    // that, queued readFrame calls would pile up and the video would
    // race ahead / lag behind indefinitely.  Drop redundant reads.
    if (m_reading.exchange(true)) return;
    if (!m_source || !m_source->isOpened()) {
        m_reading = false;
        return;
    }
    cv::Mat frame;
    int64_t frameIndex = 0;
    const IFrameSource::ReadResult result = m_source->read(frame, &frameIndex);
    m_reading = false;

    switch (result) {
        case IFrameSource::ReadResult::Ok:
            m_lastFrameIndex = static_cast<int>(frameIndex);
            // Emit the raw BGR frame.  Downstream consumers (cvMatToQImage /
            // detection) perform the single BGR->RGB conversion; converting
            // here would double-swap the channels and corrupt colors.
            emit frameReady(frame, static_cast<int>(frameIndex));
            break;
        case IFrameSource::ReadResult::NoFrame:
            // Backend (mpv) is clock-driven and has nothing new yet — the
            // pipeline keeps ticking and will pick the frame up next call.
            break;
        case IFrameSource::ReadResult::Eof:
            emit frameReadFailed();
            break;
    }
}

void VideoFrameReader::seekToFrame(int frameIndex) {
    if (m_source) m_source->seekToFrame(frameIndex);
}

void VideoFrameReader::setPaused(bool paused) {
    if (m_source) m_source->setPaused(paused);
}

void VideoFrameReader::setPlaybackSpeed(double speed) {
    if (m_source) m_source->setSpeed(speed);
}

void VideoFrameReader::startClockReading(int intervalMs) {
    // Slot runs on the reader thread, so the timer lives and fires there —
    // decode stays off the UI thread and open/seek/release slots remain
    // deliverable between ticks.
    if (!m_clockTimer) {
        m_clockTimer = new QTimer(this);
        m_clockTimer->setTimerType(Qt::CoarseTimer);
        connect(m_clockTimer, &QTimer::timeout, this,
                &VideoFrameReader::readFrame);
    }
    m_clockTimer->setInterval(std::max(1, intervalMs));
    if (!m_clockTimer->isActive()) m_clockTimer->start();
}

void VideoFrameReader::stopClockReading() {
    if (m_clockTimer) m_clockTimer->stop();
}

void VideoFrameReader::setClockInterval(int intervalMs) {
    if (m_clockTimer) m_clockTimer->setInterval(std::max(1, intervalMs));
}

void VideoFrameReader::release() {
    if (m_source) m_source->release();
}

#endif  // HAS_OPENCV_FACE_CAPTURE
