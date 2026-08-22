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

#include "YOLOModelCatalog.h"

struct aicore_cancel_token;
struct aicore_yolo_ctx;

/** Background YOLO inference worker (single image). The task follows the
 *  model: a detect GGUF emits resultReady (boxes), a depth GGUF emits
 *  depthResultReady (metric depth map). The model context is created inside
 *  run() on the worker thread; it is released on the main thread via
 *  releaseContextOnMainThread() so GPU teardown never races the render
 *  thread. */
class YOLOWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = QStringLiteral("auto");
        float confThres = 0.25f;
        float iouThres = 0.7f;
        uint32_t topK = 300;
    };

    explicit YOLOWorker(const Settings& settings, QObject* parent = nullptr);
    ~YOLOWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const YOLORunResult& result);
    void depthResultReady(const YOLODepthResult& result);
    void taskFinished(bool success);
    void modelInfoReady(const QString& info);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    bool runDetect(const QImage& rgb, const uchar* rgbData);
    bool runSegment(const QImage& rgb, const uchar* rgbData);
    bool runDepth(const QImage& rgb, const uchar* rgbData);
#endif

    Settings m_settings;
    struct aicore_yolo_ctx* m_pendingCtx = nullptr;
    aicore_cancel_token* m_cancelToken = nullptr;
};
