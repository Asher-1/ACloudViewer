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
    double preprocessMs = 0, inferenceMs = 0, postprocessMs = 0, e2eMs = 0;
    QString resolvedDevice;
    QString precision;
    QImage annotatedImage;  // input with the result boxes rendered (Qt)
};

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
        int maxNewTokens = 2048;
        QString inputPath;
        QString task = QStringLiteral("detection");
        QString text;
        QString lang = QStringLiteral("en");
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
};
