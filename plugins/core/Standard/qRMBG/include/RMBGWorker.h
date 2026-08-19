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

#include "RMBGModelCatalog.h"

struct aicore_cancel_token;
struct aicore_rmbg_ctx;

/** Background RMBG-2.0 removal worker (single image).
 *  The model context is created inside run() on the worker thread; it is
 *  released on the main thread via releaseContextOnMainThread() so GPU
 *  teardown never races the render thread. */
class RMBGWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = QStringLiteral("auto");
        /** Pixels with alpha below this value become fully transparent
         *  (0.0 disables the threshold pass). */
        float alphaThreshold = 0.5f;
    };

    explicit RMBGWorker(const Settings& settings, QObject* parent = nullptr);
    ~RMBGWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const RMBGRunResult& result);
    void taskFinished(bool success);
    void modelInfoReady(const QString& info);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
#endif

    Settings m_settings;
    struct aicore_rmbg_ctx* m_pendingCtx = nullptr;
    aicore_cancel_token* m_cancelToken = nullptr;
};
