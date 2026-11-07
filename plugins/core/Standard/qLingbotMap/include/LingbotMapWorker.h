// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QMutex>
#include <QStringList>
#include <QThread>
#include <QVector>
#include <QWaitCondition>
#include <atomic>

struct aicore_lingbot_ctx;
struct aicore_lingbot_result;

namespace lingbot_stitch {
struct WindowData; /**< official windowed math (LingbotWindowStitcher.h) */
}

/** Per-frame streaming reconstruction result (processed-resolution layout).
 *  All buffers are frame-owned copies taken inside the engine callback. */
struct LingbotFrameResult {
    QVector<float> depth;      /**< [height * width], meters */
    QVector<float> depthConf;  /**< [height * width] */
    QVector<float> c2w;        /**< 4x4 row-major camera-to-world */
    QVector<float> intrinsics; /**< [fx, fy, cx, cy] */
    QVector<float> poseEnc;    /**< 9 floats: [t, quat_xyzw, fov_y, fov_x] */
    int width = 0;
    int height = 0;
    int globalIndex = 0;      /**< index in the full source sequence */
    int windowIndex = 0;      /**< owning window (windowed mode) */
    QString sourceFile;       /**< input image this frame came from */
    QImage frameRgb;          /**< processed-resolution RGB (for colors) */
    QVector<uint8_t> skyKeep; /**< empty when sky masking is off; 255 = keep */
};

/** Compact per-frame live preview emitted while the engine streams. Points
 *  are confidence-filtered, stride-subsampled world coordinates (the same
 *  geometry the official StreamingViewer shows) — in the owning window's
 *  coordinate frame for windowed runs. */
struct LingbotFramePreview {
    int globalIndex = 0;
    int windowIndex = 0;
    int windowCount = 0;
    int width = 0;             /**< processed width (camera intrinsics) */
    int height = 0;            /**< processed height */
    QVector<float> c2w;        /**< 16 floats, row-major camera-to-world */
    QVector<float> intrinsics; /**< [fx, fy, cx, cy] */
    QVector<float> points;     /**< [n * 3] world XYZ of kept points */
    QVector<uint8_t> colors;   /**< [n * 3] RGB of kept points */
    bool degenerate = false;   /**< non-finite depth/pose: frustum only */
};

/** Per-window ALIGNED live preview emitted when a window finishes: the
 *  window's contribution frames (overlap de-duplicated) after the official
 *  similarity warp, so every batch shares the first window's coordinate
 *  frame. The main thread appends these into ONE persistent cloud and adds
 *  one camera frustum per contributed frame (also aligned) — the live map
 *  grows across window boundaries instead of being hidden when the next
 *  window-local streaming preview would take over. */
struct LingbotAlignedPreview {
    int windowIndex = 0;
    int width = 0;           /**< processed width (camera intrinsics) */
    int height = 0;          /**< processed height */
    QVector<float> points;   /**< [n * 3] world XYZ (aligned frame) */
    QVector<uint8_t> colors; /**< [n * 3] RGB of kept points */
    QVector<float> camC2w;   /**< [k * 16] aligned row-major c2w per
                                  contributed frame */
    QVector<float> camIntr;  /**< [k * 4] fx, fy, cx, cy per frame */
    QVector<int> camFrames;  /**< k global frame indices */
};

/** Aggregated result handed back to the main thread. */
struct LingbotRunResult {
    QVector<LingbotFrameResult> frames;
    QString modelFile;
    QString device;
    qint64 elapsedMs = 0;
    bool windowed = false; /**< long-sequence windowed reconstruction */
    int windowCount = 0;   /**< 1 for streaming runs */
};

/** Background LingBot-Map streaming worker.
 *
 *  The AICore context is created inside run() on the worker thread and is
 *  released on the main thread via releaseContextOnMainThread() so GPU
 *  teardown never races the render thread (same pattern as qGKD/qSAM3).
 *  The worker owns preprocessing (official crop), stream submission, and
 *  per-frame result materialization; the main thread only renders DB
 *  entities from the handed-over result.
 *
 *  Long sequences use the official windowed pipeline (keyframe_interval=1):
 *  every window runs the validated streaming primitive over a fresh KV cache
 *  (aicore_lingbot_stream_reset), consecutive windows are similarity-aligned
 *  on the overlap (LingbotWindowStitcher, port of the official numpy math)
 *  and stitched into the first window's coordinate frame. Window results stay
 *  resident until the stitch — same memory order as the streaming result. */
class LingbotMapWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;   /**< LingBot-Map GGUF */
        QString imageFolder; /**< ordered image sequence */
        /** Explicit ordered paths for headless callers. Non-empty replaces
         *  imageFolder while videoPath retains upstream highest precedence. */
        QStringList inputPaths;
        /** Upstream --video_path parity input (requires OpenCV at build
         *  time); when non-empty it replaces imageFolder, mirroring the
         *  upstream ggml_demo precedence. */
        QString videoPath;
        int videoFps = 10; /**< --fps sampling rate for --video_path */
        /** Comma-separated image extensions (upstream --image_ext);
         *  matching is case-insensitive, empty keeps the full default
         *  set (.jpg/.png/.jpeg/.bmp/.tif/.tiff). */
        QString imageExt = QStringLiteral(".jpg,.png,.jpeg,.bmp,.tif,.tiff");
        int maxFrames = 0; /**< 0 = all frames in the folder */
        int threads = 0;
        QString device = QStringLiteral("auto");
        int image_size = 518;       /**< official crop width */
        float confThreshold = 1.5f; /**< visibility confidence filter */
        /** Official --downsample_factor (fused viewer display default 10):
         *  per-frame clouds keep every Nth filtered point (parse_pc_data's
         *  sequential arange stride). 1 = full density. */
        int downsampleFactor = 10;
        /** Live-preview camera behavior (official StreamingViewer parity):
         *  0 = Global view (user-controlled, never touched);
         *  1 = Fit latest frame (the official _fit_view: mean center,
         *  fixed diagonal eye, re-fit at frames 1/10/50/120);
         *  2 = Follow latest camera (the view rides the newest frustum). */
        int liveViewMode = 1;
        /** Sky masking mode: none, native (skyseg GGUF), or cached PNG
         *  masks (upstream <scene>_sky_masks cache semantics; PNG value
         *  255 = keep, named after the source frame stem). */
        enum class SkySource { None, Native, CachedMasks };
        SkySource skySource = SkySource::None;
        QString skysegModelPath; /**< Native mode: skyseg GGUF path */
        QString skyMaskDir;      /**< CachedMasks mode: PNG mask directory */
        /** Upstream ggml_demo parity options. */
        int kvScale = 8;   /**< --kv_cache_scale (persistent scale frames) */
        int kvWindow = 64; /**< --kv_cache_window (sliding window) */
        /** Official long-stream keyframe policy (demo.py
         *  --keyframe_interval): every N-th streaming frame persists its KV;
         *  0 = auto (ceil(N/320) per the official streaming rule, 1 when the
         *  stream is shorter). Windowed mode always runs per-window
         *  keyframe_interval=1. */
        int keyframeInterval = 0;
        int frameStride = 1; /**< --stride (sample every Nth frame) */
        bool rotateClockwise90 = false; /**< --rotate_clockwise_90 */
        bool addResultToDb = true;
        /** Reconstruction mode: single persistent KV cache, or the official
         *  windowed pipeline for long sequences. */
        enum class Mode { Streaming, Windowed };
        Mode mode = Mode::Streaming;
        int windowSize = 64;   /**< keyframes per window (official default) */
        int overlap = 16;      /**< overlap frames (official default) */
        int previewStride = 4; /**< live-preview subsample (official default) */
    };

    explicit LingbotMapWorker(const Settings& settings,
                              QObject* parent = nullptr);
    ~LingbotMapWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel() {
        m_cancelRequested.store(true);
        // A paused stream waits on this condition; cancellation must wake
        // it or the worker would hang inside the frame callback.
        m_pauseCond.wakeAll();
    }
    /** Live control (official viewer pause semantics): the stream waits
     *  INSIDE the frame callback, so no engine frame is consumed while
     *  paused and the render window stops growing; resume continues the
     *  same stream. Cancellation always breaks the wait. */
    void requestPause() { m_pauseRequested.store(true); }
    void requestResume() {
        m_pauseRequested.store(false);
        m_pauseCond.wakeAll();
    }

signals:
    void logMessage(const QString& msg);
    void taskStage(const QString& stage, int percent = -1);
    void framesDecoded(int count);
    /** Live per-frame preview while the engine streams (main-thread slot). */
    void framePreviewReady(const LingbotFramePreview& preview);
    /** The processed-resolution input image of the frame just delivered
     *  (implicit-sharing copy; lets the dialog mirror the reconstruction
     *  progress like the official viewer's frame image panel). */
    void previewImage(const QImage& image, int globalIndex);
    /** Pause state transitions (emitted from the worker thread). */
    void pauseStateChanged(bool paused);
    /** ALIGNED per-window preview batch (main-thread slot): the persistent
     *  cross-window map the transient per-window previews fold into. */
    void alignedPreviewReady(const LingbotAlignedPreview& preview);
    void resultReady(const LingbotRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
    bool runInference(QString* error);
    /** Blocks inside the engine's frame callback while paused; returns
     *  when resumed or cancelled (the callback then reports cancel). */
    void waitWhilePaused();
    /** Shared per-frame engine-callback state (streaming + windowed). */
    struct StreamState {
        LingbotMapWorker* worker;
        std::vector<LingbotFrameResult>* frames; /**< stream-local results */
        const std::vector<QImage>*
                rgb;     /**< processed-res RGB per global frame */
        int total;       /**< frames in this stream call */
        int globalStart; /**< global index of local frame 0 */
        int windowIndex;
        int windowCount;
        /** Windowed mode: numeric rows land directly in the stitcher's
         *  WindowData (no intermediate O(depth-plane) copies); `frames`
         *  then carries only the sky state + identity metadata. */
        lingbot_stitch::WindowData* window = nullptr;
    };
    /** Shared streaming callback: result copy + live preview + progress. */
    static int streamCallbackEntry(void* user, const aicore_lingbot_result* r);
    bool runStreaming(aicore_lingbot_ctx* ctx,
                      const std::vector<float>& stream,
                      int frameCount,
                      int procW,
                      int procH,
                      const QStringList& files,
                      const std::vector<QImage>& rgb,
                      LingbotRunResult& result,
                      QString* error);
    bool runWindowed(aicore_lingbot_ctx* ctx,
                     const std::vector<float>& stream,
                     int frameCount,
                     int procW,
                     int procH,
                     const QStringList& files,
                     const std::vector<QImage>& rgb,
                     LingbotRunResult& result,
                     QString* error);
    /** Builds one window's cached-mask injection buffer ([start, end)). */
    bool injectCachedMasks(aicore_lingbot_ctx* ctx,
                           const QStringList& files,
                           int start,
                           int frameCount,
                           int procW,
                           int procH);

    Settings m_settings;
    std::atomic<bool> m_cancelRequested{false};
    /** Live pause state (see requestPause/requestResume). */
    std::atomic<bool> m_pauseRequested{false};
    bool m_pauseNotified =
            false; /**< dedupes pauseStateChanged, worker thread only */
    QMutex m_pauseMutex;
    QWaitCondition m_pauseCond;
    void* m_pendingCtx = nullptr; /**< aicore_lingbot_ctx* (opaque) */
    // Worker-thread-only stream state (used inside the C callback).
    friend struct StreamState;
    int m_delivered = 0;
    bool m_skyReady = false;
    QElapsedTimer m_elapsed;
    /** Running point count of the transient online preview cloud for the
     *  current window/stream; drives the adaptive preview stride that
     *  keeps the main thread's throttled re-conversion bounded. */
    size_t m_previewPointTotal = 0;
    /** Running point count of the ALIGNED cross-window preview batches;
     *  stretches their pixel stride once the persistent map cloud passes
     *  the same point budget as the transient preview. */
    size_t m_alignedPointTotal = 0;
    /** Mode actually used for the current run: Streaming upgraded to
     *  Windowed by the official long-sequence policy. Computed in run()
     *  where the stream bound is known, consumed by runInference(). */
    Settings::Mode m_effectiveMode = Settings::Mode::Streaming;
};

Q_DECLARE_METATYPE(LingbotFrameResult)
Q_DECLARE_METATYPE(LingbotFramePreview)
Q_DECLARE_METATYPE(LingbotAlignedPreview)
Q_DECLARE_METATYPE(LingbotRunResult)
