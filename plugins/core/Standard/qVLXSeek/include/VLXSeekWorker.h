// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Background VLX-Seek inference worker. The three-model context is created
// inside run() on the worker thread and released on the main thread via
// releaseContextOnMainThread() so GPU teardown never races the render
// thread. One request per run; the dialog serializes runs.

#pragma once

#include <QImage>
#include <QMetaType>
#include <QObject>
#include <QRectF>
#include <QString>
#include <QStringList>
#include <QThread>
#include <QVector>

struct aicore_vlx_seek_ctx;

struct VLXSeekRunResult {
    QString imagePath;
    QString answer;
    struct Box {
        QRectF rect;    // original-image pixels
        QString label;  // model-generated (or <objN> fallback)
    };
    QVector<Box> boxes;
    int32_t promptTokens = 0;
    int32_t completionTokens = 0;
    int32_t proposalCount = 0;
    /** Human-readable proposal source of the run ("wedetect" built-in
     *  proposer, "external" boxes JSON, or "ocr-det" text detector). */
    QString proposalSource = QStringLiteral("wedetect");
    /** Wall time of the plugin-side OCR-det proposal pass (ocr-det runs
     *  only; 0 when the proposals came from the task runtime). */
    double proposalMs = 0;
    /** Wall time of the three-GGUF context creation (0 when the context
     *  was reused). Reported in the run summary — first-run latency is
     *  dominated by the load, not the inference. */
    double modelLoadMs = 0;
    double preprocessMs = 0, inferenceMs = 0, postprocessMs = 0, e2eMs = 0;
    QString resolvedDevice;
    QString precision;
    QImage annotatedImage;  // input with the result boxes rendered (Qt)
};

// The worker thread hands results to the main thread through a queued
// connection; without this declaration Qt silently drops the signal and
// the dialog never sees a result (the qQuerySplat lesson).
Q_DECLARE_METATYPE(VLXSeekRunResult)

class VLXSeekWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString wedetectPath;
        QString visionPath;
        QString llmPath;
        /** Vision/LLM quantization of the loaded rows; reported verbatim in
         *  the model_info metadata (the paths above are the loading truth). */
        QString precision;
        QString device = QStringLiteral("auto");
        int threads = 0;
        int maxNewTokens = 1024;
        QString inputPath;
        QString task = QStringLiteral("detection");
        QString text;
        QString lang = QStringLiteral("en");
        /** Comma-separated zero-based proposal indexes consumed by the
         *  brief_region_caption / region_ocr tasks (the runtime rejects
         *  those tasks without targets); parsed by parseTargetIndexes. */
        QString targetIndexes;
        /** Optional external proposal-boxes JSON file (upstream --bbox-list
         *  format). Empty = the built-in WeDetect-Uni proposer runs first. */
        QString externalBoxesPath;
        /** Region-proposal source: "builtin" (WeDetect-Uni inside the task
         *  runtime), "external" (externalBoxesPath JSON) or "ocr_det" (the
         *  PP-OCR text detector composed from the public AICore OCR API;
         *  region_ocr only). */
        QString proposalSource = QStringLiteral("builtin");
        /** Resolved PP-OCR det/rec GGUF paths for the ocr_det proposer
         *  (ocrProposerModels + the shared downloader). */
        QString ocrDetPath;
        QString ocrRecPath;
    };

    explicit VLXSeekWorker(const Settings& settings, QObject* parent = nullptr);
    ~VLXSeekWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();

#ifdef AICore_ENABLED
    /** Run one image through a loaded context (decode, inference, official
     *  JSON). Shared by run() and the headless -VLXSEEK_RUN batch command so
     *  both paths produce identical results. \p ctx must be ready; \p
     *  annotated receives the Qt-rendered copy when non-null. Returns false
     *  with \p error set on failure. */
    static bool runImage(aicore_vlx_seek_ctx* ctx,
                         const Settings& settings,
                         VLXSeekRunResult* result,
                         QImage* annotated,
                         QString* error);
#endif

signals:
    void logMessage(const QString& msg);
    void resultReady(const VLXSeekRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
#endif

    Settings m_settings;
    struct aicore_vlx_seek_ctx* m_pendingCtx = nullptr;
    double m_loadMs = 0;
};

Q_DECLARE_METATYPE(VLXSeekWorker::Settings)
