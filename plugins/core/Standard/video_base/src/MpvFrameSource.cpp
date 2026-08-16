// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MpvFrameSource.h"

#if defined(HAS_LIBMPV) && defined(MPV_RENDER_API_TYPE_SW)

#include <cstdio>
#include <cstring>

#include <opencv2/imgproc.hpp>

namespace {
// libmpv client API (client.h) is ABI-stable since 0.33; the software
// renderer (render.h) was added in 0.35.  Guard against older headers so a
// distro without the feature still builds (available() then reports false
// and VideoFrameReader uses the OpenCV backend).
constexpr bool kHasSwRender = true;
}  // namespace

MpvFrameSource::MpvFrameSource() {
    if (!available()) return;
    m_mpv = mpv_create();
    if (!m_mpv) return;

    // Playback options — see mpv(1).  hwdec=auto picks VAAPI / D3D11 /
    // Vulkan from the OS driver (no CUDA runtime needed); hr-seek gives
    // frame-exact seeking; vd-lavc-dr=no keeps SW render compatible.
    mpv_set_option_string(m_mpv, "hwdec", "auto");
    mpv_set_option_string(m_mpv, "hr-seek", "yes");
    mpv_set_option_string(m_mpv, "vd-lavc-dr", "no");
    mpv_set_option_string(m_mpv, "audio", "auto");
    mpv_set_option_string(m_mpv, "volume", "100");
    mpv_set_option_string(m_mpv, "cache", "yes");
    mpv_set_option_string(m_mpv, "demuxer-readahead-secs", "2");
    mpv_set_option_string(m_mpv, "keep-open", "no");
    mpv_set_option_string(m_mpv, "loop-file", "no");
    mpv_set_option_string(m_mpv, "pause", "yes");  // openVideo() unpauses
    mpv_request_log_messages(m_mpv, "no");

    if (mpv_initialize(m_mpv) < 0) {
        mpv_terminate_destroy(m_mpv);
        m_mpv = nullptr;
        return;
    }
    m_thread = std::thread(&MpvFrameSource::mpvThreadMain, this);
}

MpvFrameSource::~MpvFrameSource() {
    if (m_mpv) {
        m_shutdown.store(true, std::memory_order_release);
        mpv_command_string(m_mpv, "quit");  // wake up mpv_wait_event()
        if (m_thread.joinable()) m_thread.join();
        // The render context is freed at the end of mpvThreadMain() (same
        // thread that created it), so only the handle remains.
        mpv_terminate_destroy(m_mpv);
        m_mpv = nullptr;
    }
}

bool MpvFrameSource::available() {
    return kHasSwRender;
}

bool MpvFrameSource::isOpened() const {
    return m_opened.load(std::memory_order_acquire);
}

bool MpvFrameSource::openVideo(const std::string& path, int /*backendHint*/) {
    release();
    if (!m_mpv || !kHasSwRender) return false;

    {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        m_opened = false;
        m_loadFailed = false;
        m_eof = false;
        m_latestFrameIndex = -1;
        m_consumedIndex = -1;
        m_swWidth = m_swHeight = 0;
    }
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_latestFrame.release();
    }

    const char* cmd[] = {"loadfile", path.c_str(), "replace", nullptr};
    if (mpv_command(m_mpv, cmd) < 0) return false;

    // Block until the mpv thread signals FILE_LOADED (or an error).
    std::unique_lock<std::mutex> lock(m_stateMutex);
    m_stateCv.wait_for(lock, std::chrono::seconds(10),
                       [this]() { return m_opened || m_loadFailed; });
    return m_opened;
}

IFrameSource::ReadResult MpvFrameSource::read(cv::Mat& out,
                                              int64_t* frameIndex) {
    if (!m_opened.load(std::memory_order_acquire)) return ReadResult::NoFrame;
    if (m_eof.load(std::memory_order_acquire)) return ReadResult::Eof;

    std::lock_guard<std::mutex> lock(m_frameMutex);
    if (m_latestFrame.empty() || m_latestFrameIndex < 0) {
        return ReadResult::NoFrame;
    }
    if (m_latestFrameIndex == m_consumedIndex) return ReadResult::NoFrame;
    m_consumedIndex = m_latestFrameIndex;
    if (frameIndex) *frameIndex = m_latestFrameIndex;
    // bgr0 (BGRA) -> BGR, matching the OpenCV backend's frame format.
    cv::cvtColor(m_latestFrame, out, cv::COLOR_BGRA2BGR);
    return ReadResult::Ok;
}

bool MpvFrameSource::seekToFrame(int frameIndex) {
    const double fps = m_fps.load(std::memory_order_acquire);
    if (!m_opened || fps <= 0) return false;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f",
                  static_cast<double>(frameIndex) / fps);
    const char* cmd[] = {"seek", buf, "absolute", nullptr};
    if (mpv_command(m_mpv, cmd) < 0) return false;
    // A seek clears the EOF state; force the next rendered frame to be
    // handed out even if the slider landed on the same position.
    m_eof.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_latestFrameIndex = -1;
        m_consumedIndex = -1;
    }
    return true;
}

int64_t MpvFrameSource::frameCount() const {
    return m_frameCount.load(std::memory_order_acquire);
}

double MpvFrameSource::fps() const {
    return m_fps.load(std::memory_order_acquire);
}

int MpvFrameSource::width() const {
    return m_width.load(std::memory_order_acquire);
}

int MpvFrameSource::height() const {
    return m_height.load(std::memory_order_acquire);
}

void MpvFrameSource::release() {
    if (m_mpv) mpv_command_string(m_mpv, "stop");
    m_opened = false;
    m_eof.store(false, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_latestFrame.release();
        m_latestFrameIndex = -1;
        m_consumedIndex = -1;
    }
}

bool MpvFrameSource::setPaused(bool paused) {
    if (!m_mpv) return false;
    return mpv_set_property_string(m_mpv, "pause", paused ? "yes" : "no") >= 0;
}

bool MpvFrameSource::setSpeed(double speed) {
    if (!m_mpv || speed <= 0.0) return false;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.6f", speed);
    return mpv_set_property_string(m_mpv, "speed", buf) >= 0;
}

void MpvFrameSource::onRenderUpdate(void* ctx) {
    auto* self = static_cast<MpvFrameSource*>(ctx);
    self->m_renderPending.store(true, std::memory_order_release);
}

void MpvFrameSource::mpvThreadMain() {
    // The render context must be created, rendered and freed on this same
    // thread (libmpv constraint for the render API).
    mpv_render_param initParams[] = {
            {MPV_RENDER_PARAM_API_TYPE, const_cast<char*>(MPV_RENDER_API_TYPE_SW)},
            {0, nullptr},
    };
    if (mpv_render_context_create(&m_renderCtx, m_mpv, initParams) < 0) {
        m_renderCtx = nullptr;
    } else {
        mpv_render_context_set_update_callback(m_renderCtx, onRenderUpdate,
                                               this);
    }

    while (!m_shutdown.load(std::memory_order_acquire)) {
        mpv_event* ev = mpv_wait_event(m_mpv, 0.005);
        if (ev && ev->event_id == MPV_EVENT_SHUTDOWN) break;
        if (ev) onMpvEvent(ev);

        if (m_renderCtx &&
            (m_renderPending.exchange(false) ||
             mpv_render_context_update(m_renderCtx) > 0)) {
            renderFrame();
        }
    }

    if (m_renderCtx) {
        mpv_render_context_free(m_renderCtx);
        m_renderCtx = nullptr;
    }
}

void MpvFrameSource::onMpvEvent(mpv_event* ev) {
    switch (ev->event_id) {
        case MPV_EVENT_FILE_LOADED: {
            updateMetadata();
            m_opened = true;
            m_eof = false;
            // Start A/V playback right away (the widget drains frames at
            // its own pace via read()).
            mpv_set_property_string(m_mpv, "pause", "no");
            std::lock_guard<std::mutex> lock(m_stateMutex);
            m_stateCv.notify_all();
            break;
        }
        case MPV_EVENT_END_FILE: {
            const auto* end =
                    static_cast<mpv_event_end_file*>(ev->data);
            if (end->reason == MPV_END_FILE_REASON_EOF) {
                // Ignore stale END_FILE events queued before a seek: the
                // eof-reached property reflects the actual playback state.
                char* eofProp = mpv_get_property_string(m_mpv, "eof-reached");
                const bool atEof =
                        eofProp && std::strcmp(eofProp, "yes") == 0;
                mpv_free(eofProp);
                if (atEof) m_eof.store(true, std::memory_order_release);
            } else if (end->reason == MPV_END_FILE_REASON_ERROR) {
                m_loadFailed = true;
                std::lock_guard<std::mutex> lock(m_stateMutex);
                m_stateCv.notify_all();
            }
            break;
        }
        default:
            break;
    }
}

void MpvFrameSource::renderFrame() {
    const int64_t w = mpv_get_property_int(m_mpv, "video-params/w");
    const int64_t h = mpv_get_property_int(m_mpv, "video-params/h");
    if (w <= 0 || h <= 0) return;

    if (m_swWidth != w || m_swHeight != h) {
        m_swBuffer.assign(static_cast<size_t>(w) * h * 4, 0);
        m_swWidth = static_cast<int>(w);
        m_swHeight = static_cast<int>(h);
    }

    void* bufPtr = m_swBuffer.data();
    int swSize[2] = {static_cast<int>(w), static_cast<int>(h)};
    int swStride[2] = {static_cast<int>(w * 4), 0};
    mpv_render_param params[] = {
            {MPV_RENDER_PARAM_SW_SIZE, swSize},
            {MPV_RENDER_PARAM_SW_FORMAT, const_cast<char*>("bgr0")},
            {MPV_RENDER_PARAM_SW_STRIDE, swStride},
            {MPV_RENDER_PARAM_SW_POINTER, &bufPtr},
            {0, nullptr},
    };
    if (mpv_render_context_render(m_renderCtx, params) != 0) return;

    // video-pts distinguishes a genuinely new frame from a re-render of the
    // same frame (e.g. after a widget redraw request).
    const double pts = mpv_get_property_double(m_mpv, "video-pts");
    if (pts < 0.0 || pts == m_lastPts) return;
    m_lastPts = pts;

    cv::Mat bgra(static_cast<int>(h), static_cast<int>(w), CV_8UC4,
                 m_swBuffer.data());
    {
        std::lock_guard<std::mutex> lock(m_frameMutex);
        m_latestFrame = bgra.clone();
        const double fps = m_fps.load(std::memory_order_acquire);
        m_latestFrameIndex =
                fps > 0.0 ? static_cast<int64_t>(pts * fps + 0.5)
                          : m_latestFrameIndex + 1;
    }
    m_width.store(static_cast<int>(w), std::memory_order_release);
    m_height.store(static_cast<int>(h), std::memory_order_release);
}

void MpvFrameSource::updateMetadata() {
    const double duration = mpv_get_property_double(m_mpv, "duration");
    double fps = mpv_get_property_double(m_mpv, "estimated-vf-fps");
    if (fps <= 0.0) fps = mpv_get_property_double(m_mpv, "fps");
    if (fps <= 0.0) fps = 30.0;  // fallback for VFR / unknown
    m_fps.store(fps, std::memory_order_release);
    m_frameCount.store(
            duration > 0.0 ? static_cast<int64_t>(duration * fps + 0.5) : 0,
            std::memory_order_release);
}

#endif  // HAS_LIBMPV && MPV_RENDER_API_TYPE_SW

// Older libmpv (no software rendering): the header provides an
// always-unavailable stub; keep the symbol definition here for linkage.
#if defined(HAS_LIBMPV) && !defined(MPV_RENDER_API_TYPE_SW)
bool MpvFrameSource::available() { return false; }
#endif
