// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OpenCVFrameSource.h"

#ifdef HAS_OPENCV_FACE_CAPTURE

#include <opencv2/core/utils/logger.hpp>

#if !defined(_WIN32)
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

// RAII guard that temporarily redirects stderr to /dev/null.
// Suppresses noisy libva / FFmpeg hw-acceleration errors on systems
// without GPU drivers (headless servers, VMs, containers).
class StderrGuard {
 public:
  StderrGuard() {
#if !defined(_WIN32)
    fflush(stderr);
    m_saved = dup(STDERR_FILENO);
    m_devnull = open("/dev/null", O_WRONLY);
    if (m_devnull >= 0) dup2(m_devnull, STDERR_FILENO);
#endif
  }
  ~StderrGuard() {
#if !defined(_WIN32)
    if (m_saved >= 0) {
      dup2(m_saved, STDERR_FILENO);
      close(m_saved);
    }
    if (m_devnull >= 0) close(m_devnull);
#endif
  }

 private:
#if !defined(_WIN32)
  int m_saved = -1;
  int m_devnull = -1;
#endif
};

// Drains decoded frames until the capture position reaches `target`.
// FFmpeg backends land on a keyframe boundary and decode forward, so VFR
// videos or B-frame reordering can land a few frames short of the target.
// Bounded so a backend that cannot report POS_FRAMES degrades to returning
// the first decoded frame.
bool readToExactFrame(cv::VideoCapture& cap, int target, cv::Mat& out) {
    if (!cap.read(out) || out.empty()) return false;
    int64_t pos = static_cast<int64_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
    int guard = 0;
    while (pos >= 0 && pos < target && guard < 30) {
        if (!cap.grab()) return false;
        pos = static_cast<int64_t>(cap.get(cv::CAP_PROP_POS_FRAMES));
        ++guard;
    }
    if (guard > 0) {
        if (!cap.retrieve(out) || out.empty()) return false;
    }
    return true;
}

}  // namespace

bool OpenCVFrameSource::openVideoWithHw(cv::VideoCapture& cap,
                                        const std::string& path, int backend) {
    // Best-effort hardware-accelerated decode: OpenCV >= 4.5.2 exposes
    // CAP_PROP_HW_ACCELERATION (VAAPI on Linux, D3D11 on Windows — both
    // provided by the OS/driver, so no extra runtime to ship).  When the
    // driver or codec is unsupported, OpenCV falls back to software
    // internally; older OpenCV builds skip this block entirely.
#if defined(CV_VERSION_MAJOR) && \
    (CV_VERSION_MAJOR > 4 || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 5))
    if (backend == cv::CAP_FFMPEG || backend == cv::CAP_ANY) {
        // Suppress libva / FFmpeg hw-acceleration stderr noise — on
        // systems without VAAPI/D3D11 drivers the probe prints errors
        // that are harmless (OpenCV falls back to software decode).
        StderrGuard guard;
        cap.open(path, backend, {cv::CAP_PROP_HW_ACCELERATION,
                                 cv::VIDEO_ACCELERATION_ANY});
        if (cap.isOpened()) return true;
    }
#else
    (void)cap;
    (void)path;
    (void)backend;
    return false;
#endif
    return false;
}

bool OpenCVFrameSource::openVideo(const std::string& path, int backendHint) {
    release();
    m_seekPendingFrame = -1;
    const int backend = backendHint != 0 ? backendHint : cv::CAP_ANY;
    if (!openVideoWithHw(m_cap, path, backend)) {
        m_cap.open(path, backend);
        if (!m_cap.isOpened()) m_cap.open(path, cv::CAP_ANY);
    }
    if (m_cap.isOpened()) {
        // Small decoder-side buffer: keeps seek + read aligned and avoids
        // the pipeline racing ahead of the displayed frame.
        m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    return m_cap.isOpened();
}

bool OpenCVFrameSource::openCamera(int deviceIndex, int backendHint) {
    release();
    m_cap.open(deviceIndex, backendHint);
    if (m_cap.isOpened()) {
        m_cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        m_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        m_cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
    }
    return m_cap.isOpened();
}

bool OpenCVFrameSource::isOpened() const {
    return m_cap.isOpened();
}

IFrameSource::ReadResult OpenCVFrameSource::read(cv::Mat& out,
                                                 int64_t* frameIndex) {
    if (!m_cap.isOpened()) return ReadResult::Eof;

    cv::Mat frame;
    bool ok = false;

    // Exact-seek alignment: after seekToFrame() the FFmpeg backend lands on
    // the keyframe boundary and decodes forward; VFR timestamps or B-frame
    // reordering can leave the reported position a few frames short of the
    // target.  Drain grab() calls until the position catches up (bounded so
    // a backend that cannot report POS_FRAMES degrades to the old behavior).
    if (m_seekPendingFrame >= 0) {
        ok = readToExactFrame(m_cap, m_seekPendingFrame, frame);
        m_seekPendingFrame = -1;  // alignment satisfied (or best-effort)
    } else {
        ok = m_cap.read(frame) && !frame.empty();
    }

    if (!ok) return ReadResult::Eof;
    out = frame;  // BGR, shallow (refcounted) — pipeline swaps ownership
    if (frameIndex) {
        *frameIndex = static_cast<int64_t>(m_cap.get(cv::CAP_PROP_POS_FRAMES));
    }
    return ReadResult::Ok;
}

bool OpenCVFrameSource::seekToFrame(int frameIndex) {
    if (!m_cap.isOpened()) return false;
    m_cap.set(cv::CAP_PROP_POS_FRAMES, frameIndex);
    // read() drains decoded frames until this index is reached.
    m_seekPendingFrame = frameIndex;
    return true;
}

int64_t OpenCVFrameSource::frameCount() const {
    return m_cap.isOpened()
                   ? static_cast<int64_t>(m_cap.get(cv::CAP_PROP_FRAME_COUNT))
                   : 0;
}

double OpenCVFrameSource::fps() const {
    return m_cap.isOpened() ? m_cap.get(cv::CAP_PROP_FPS) : 0.0;
}

int OpenCVFrameSource::width() const {
    return m_cap.isOpened()
                   ? static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_WIDTH))
                   : 0;
}

int OpenCVFrameSource::height() const {
    return m_cap.isOpened()
                   ? static_cast<int>(m_cap.get(cv::CAP_PROP_FRAME_HEIGHT))
                   : 0;
}

void OpenCVFrameSource::release() {
    if (m_cap.isOpened()) m_cap.release();
    m_seekPendingFrame = -1;
}

#endif  // HAS_OPENCV_FACE_CAPTURE
