// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
// MpvFrameSource: IFrameSource implementation backed by libmpv (the same
// playback engine that powers mpv / IINA).
//
// Capabilities over the OpenCV path:
//  - audio output (mpv's own audio API: ALSA/PulseAudio/WASAPI/CoreAudio)
//  - hardware decode via hwdec=auto (VAAPI on Linux, D3D11 on Windows,
//    Vulkan where available — all provided by the OS/driver, no CUDA
//    runtime to ship)
//  - frame-exact seeking (hr-seek)
//  - built-in demuxer read-ahead cache
//
// Frames are delivered to the CPU through mpv's software rendering API
// (MPV_RENDER_API_TYPE_SW, mpv >= 0.35), so the inference pipeline keeps
// receiving plain BGR cv::Mat without any OpenGL context.  An internal
// thread runs the mpv event loop + SW renderer; read() is a non-blocking
// "latest frame" pull guarded by a mutex.
//
// When the libmpv headers are older than 0.35 (no MPV_RENDER_API_TYPE_SW)
// the class compiles to an always-unavailable stub and VideoFrameReader
// silently falls back to OpenCVFrameSource — builds never fail.
// ----------------------------------------------------------------------------

#pragma once

#include "IFrameSource.h"

#ifdef HAS_LIBMPV
#include <mpv/client.h>
#include <mpv/render.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

// The software-rendering API (MPV_RENDER_API_TYPE_SW) landed in libmpv
// 0.35; older headers lack both the macro and the implementation.  The
// class then compiles to an always-unavailable stub so VideoFrameReader
// silently falls back to OpenCVFrameSource — builds never fail.
#ifdef MPV_RENDER_API_TYPE_SW

class MpvFrameSource : public IFrameSource {
public:
    MpvFrameSource();
    ~MpvFrameSource() override;

    // True when this build can actually render frames (libmpv >= 0.35 with
    // the software-rendering API).
    static bool available();

    bool openVideo(const std::string& path, int backendHint = 0) override;
    bool openCamera(int deviceIndex, int backendHint = 0) override {
        (void)deviceIndex;
        (void)backendHint;
        return false;  // camera input is OpenCV-only
    }

    bool isOpened() const override;
    ReadResult read(cv::Mat& out, int64_t* frameIndex) override;
    bool seekToFrame(int frameIndex) override;

    int64_t frameCount() const override;
    double fps() const override;
    int width() const override;
    int height() const override;

    void release() override;

    bool setPaused(bool paused) override;
    bool setSpeed(double speed) override;

private:
    // Invoked by mpv on its internal thread whenever a new video frame is
    // ready to be rendered.
    static void onRenderUpdate(void* ctx);

    void mpvThreadMain();
    void onMpvEvent(mpv_event* ev);
    void renderFrame();  // mpv thread only (render context lives there)
    void updateMetadata();

    mpv_handle* m_mpv = nullptr;
    // Created / rendered / freed exclusively on m_thread (libmpv requires
    // all render-context calls on the thread that created it).
    mpv_render_context* m_renderCtx = nullptr;
    std::thread m_thread;

    std::atomic<bool> m_shutdown{false};
    std::atomic<bool> m_renderPending{false};  // set by onRenderUpdate
    std::atomic<bool> m_opened{false};
    std::atomic<bool> m_loadFailed{false};
    std::atomic<bool> m_eof{false};
    std::atomic<int> m_width{0};
    std::atomic<int> m_height{0};

    // Guarded by m_stateMutex: openVideo() blocks until FILE_LOADED /
    // load failure is signalled by the mpv thread.
    std::mutex m_stateMutex;
    std::condition_variable m_stateCv;

    // Latest rendered frame (BGRA, bgr0) + its frame number.  Guarded by
    // m_frameMutex; written by the mpv thread, read by the reader thread.
    std::mutex m_frameMutex;
    cv::Mat m_latestFrame;
    int64_t m_latestFrameIndex = -1;
    int64_t m_consumedIndex = -1;  // last frame handed out by read()

    // mpv-thread-only state.
    std::vector<uint8_t> m_swBuffer;
    int m_swWidth = 0;
    int m_swHeight = 0;
    double m_lastPts = -1.0;  // video-pts of the last published frame

    // Written by the mpv thread (updateMetadata), read by the reader thread
    // (read / seek / metadata accessors).
    std::atomic<double> m_fps{0.0};
    std::atomic<int64_t> m_frameCount{0};
};

#else  // !MPV_RENDER_API_TYPE_SW

// libmpv too old for software rendering: report unavailable.
class MpvFrameSource : public IFrameSource {
public:
    static bool available() { return false; }
    bool openVideo(const std::string&, int = 0) override { return false; }
    bool openCamera(int, int = 0) override { return false; }
    bool isOpened() const override { return false; }
    ReadResult read(cv::Mat&, int64_t*) override {
        return ReadResult::NoFrame;
    }
    bool seekToFrame(int) override { return false; }
    int64_t frameCount() const override { return 0; }
    double fps() const override { return 0.0; }
    int width() const override { return 0; }
    int height() const override { return 0; }
    void release() override {}
};

#endif  // MPV_RENDER_API_TYPE_SW

#endif  // HAS_LIBMPV
