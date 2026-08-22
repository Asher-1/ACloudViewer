// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QObject>
#include <QString>
#include <QThread>
#include <QVector>

#include <cstdint>

#include "RFDetrModelCatalog.h"

struct aicore_cancel_token;
struct aicore_rfdetr_ctx;

/** Background RF-DETR inference worker (single image or folder batch).
 *  The model context is created inside run() on the worker thread; it is
 *  released on the main thread via releaseContextOnMainThread() so GPU
 *  teardown never races the render thread. */
class RFDetrWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = QStringLiteral("auto");
        float threshold = 0.5f;
        uint32_t topK = 300;
        /** Class allowlist (empty = detect all classes the model knows).
         *  Applied by the engine's post-processing, so filtered classes
         *  never appear in the result JSON. */
        QVector<uint32_t> classFilter;
    };

    explicit RFDetrWorker(const Settings& settings, QObject* parent = nullptr);
    ~RFDetrWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const RFDetrRunResult& result);
    void taskFinished(bool success);
    void modelInfoReady(const QString& info);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
#endif

    Settings m_settings;
    struct aicore_rfdetr_ctx* m_pendingCtx = nullptr;
    aicore_cancel_token* m_cancelToken = nullptr;
};
