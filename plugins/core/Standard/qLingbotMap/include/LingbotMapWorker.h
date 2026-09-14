// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QElapsedTimer>
#include <QImage>
#include <QStringList>
#include <QThread>
#include <QVector>
#include <atomic>

struct aicore_lingbot_ctx;

/** Per-frame streaming reconstruction result (processed-resolution layout).
 *  All buffers are frame-owned copies taken inside the engine callback. */
struct LingbotFrameResult {
    QVector<float> depth;      /**< [height * width], meters */
    QVector<float> depthConf;  /**< [height * width] */
    QVector<float> c2w;        /**< 4x4 row-major camera-to-world */
    QVector<float> intrinsics; /**< [fx, fy, cx, cy] */
    int width = 0;
    int height = 0;
    QString sourceFile;       /**< input image this frame came from */
    QImage frameRgb;          /**< processed-resolution RGB (for colors) */
    QVector<uint8_t> skyKeep; /**< empty when sky masking is off; 255 = keep */
};

/** Aggregated result handed back to the main thread. */
struct LingbotRunResult {
    QVector<LingbotFrameResult> frames;
    QString modelFile;
    QString device;
    qint64 elapsedMs = 0;
};

/** Background LingBot-Map streaming worker.
 *
 *  The AICore context is created inside run() on the worker thread and is
 *  released on the main thread via releaseContextOnMainThread() so GPU
 *  teardown never races the render thread (same pattern as qGKD/qSAM3).
 *  The worker owns preprocessing (official crop), stream submission, and
 *  per-frame result materialization; the main thread only renders DB
 *  entities from the handed-over result. */
class LingbotMapWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;   /**< LingBot-Map GGUF */
        QString imageFolder; /**< ordered image sequence */
        int maxFrames = 0;   /**< 0 = all frames in the folder */
        int threads = 0;
        QString device = QStringLiteral("auto");
        int image_size = 518;       /**< official crop width */
        float confThreshold = 1.5f; /**< visibility confidence filter */
        /** Sky masking mode: none, native (skyseg GGUF), or cached PNG
         *  masks (upstream <scene>_sky_masks cache semantics; PNG value
         *  255 = keep, named after the source frame stem). */
        enum class SkySource { None, Native, CachedMasks };
        SkySource skySource = SkySource::None;
        QString skysegModelPath; /**< Native mode: skyseg GGUF path */
        QString skyMaskDir;      /**< CachedMasks mode: PNG mask directory */
        bool addResultToDb = true;
    };

    explicit LingbotMapWorker(const Settings& settings,
                              QObject* parent = nullptr);
    ~LingbotMapWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel() { m_cancelRequested.store(true); }

signals:
    void logMessage(const QString& msg);
    void taskStage(const QString& stage, int percent = -1);
    void framesDecoded(int count);
    void resultReady(const LingbotRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
    bool runInference(QString* error);

    Settings m_settings;
    std::atomic<bool> m_cancelRequested{false};
    void* m_pendingCtx = nullptr; /**< aicore_lingbot_ctx* (opaque) */
    // Worker-thread-only stream state (used inside the C callback).
    int m_delivered = 0;
    bool m_skyReady = false;
    QElapsedTimer m_elapsed;
};

Q_DECLARE_METATYPE(LingbotFrameResult)
Q_DECLARE_METATYPE(LingbotRunResult)
