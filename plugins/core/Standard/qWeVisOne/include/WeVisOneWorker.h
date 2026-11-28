// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Background WeVisOne-VL inference worker. The three-model context is
// created inside run() on the worker thread and released on the main thread
// via releaseContextOnMainThread() so GPU teardown never races the render
// thread. One request per run; the dialog serializes runs.

#pragma once

#include <QImage>
#include <QMetaType>
#include <QObject>
#include <QRectF>
#include <QString>
#include <QThread>
#include <QVector>

struct aicore_wevisone_ctx;

struct WeVisOneRunResult {
    QString imagePath;
    QString answer;
    QVector<QRectF> boxes;  // original-image pixels (unlabeled)
    int32_t promptTokens = 0;
    int32_t completionTokens = 0;
    /** True when the answer carried <OBJ> tokens and the pixel decoder
     *  ran (segmentation / matting / freeform-with-masks answers). */
    bool hasMasks = false;
    int32_t matteCount = 0;
    /** First decoded matte (the upstream RunResult retains only matte 0):
     *  row-major f32 alpha in [0,1], matteWidth * matteHeight floats. */
    int32_t matteWidth = 0;
    int32_t matteHeight = 0;
    QVector<float> matte;
    /** Wall time of the three-GGUF context creation. Reported in the run
     *  summary — first-run latency is dominated by the load. */
    double modelLoadMs = 0;
    double preprocessMs = 0, inferenceMs = 0, postprocessMs = 0, e2eMs = 0;
    QString resolvedDevice;
    QString precision;
    QImage annotatedImage;  // input with boxes + matte overlay (Qt)
};

// The worker thread hands results to the main thread through a queued
// connection; without this declaration Qt silently drops the signal and
// the dialog never sees a result (the qQuerySplat lesson).
Q_DECLARE_METATYPE(WeVisOneRunResult)

class WeVisOneWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString vlmPath;
        QString mmprojPath;
        QString decoderPath;
        /** Checkpoint family of the loaded rows ("2b"/"4b"). */
        QString size = QStringLiteral("2b");
        /** VLM quantization of the loaded rows; reported verbatim in the
         *  model_info metadata (the paths above are the loading truth). */
        QString precision = QStringLiteral("q8_0");
        /** Pixel-decoder weight tier ("f32" parity-gated default, "f16"). */
        QString decoderPrecision = QStringLiteral("f32");
        QString device = QStringLiteral("auto");
        int threads = 0;
        int maxNewTokens = 4096;
        /** Run the pixel decoder when the answer carries <OBJ> tokens. */
        bool decodeMasks = true;
        QString inputPath;
        QString task = QStringLiteral("detection");
        /** Newline-separated category list, referring phrase or freeform
         *  instruction (ignored by the fixed ocr prompt). */
        QString query;
        /** visual_prompt reference boxes, upstream --ref-boxes spec
         *  ("x1,y1,x2,y2;..." in original-image pixels). */
        QString refBoxes;
    };

    explicit WeVisOneWorker(const Settings& settings,
                            QObject* parent = nullptr);
    ~WeVisOneWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();

#ifdef AICore_ENABLED
    /** Run one image through a loaded context (decode, inference, typed
     *  getters). Shared by run() and the headless -WEVISONE_RUN batch
     *  command so both paths produce identical results. \p ctx must be
     *  ready; \p annotated receives the Qt-rendered copy when non-null.
     *  Returns false with \p error set on failure. */
    static bool runImage(aicore_wevisone_ctx* ctx,
                         const Settings& settings,
                         WeVisOneRunResult* result,
                         QImage* annotated,
                         QString* error);
#endif

signals:
    void logMessage(const QString& msg);
    void resultReady(const WeVisOneRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
#endif

    Settings m_settings;
    struct aicore_wevisone_ctx* m_pendingCtx = nullptr;
};

Q_DECLARE_METATYPE(WeVisOneWorker::Settings)