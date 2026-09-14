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
#include <QStringList>
#include <QThread>
#include <QVector>

#include "GKDModelCatalog.h"

struct aicore_cancel_token;
struct aicore_gkd_ctx;

/** Background GKDT keypoint detection worker.
 *
 *  The GKD context (and the optional YOLO-World detector context for
 *  multi-object mode) is created inside run() on the worker thread and is
 *  released on the main thread via releaseContextOnMainThread() so GPU
 *  teardown never races the render thread. Cancellation is cooperative via
 *  a caller-owned aicore_cancel_token. */
class GKDWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath; /**< GKD GGUF */
        QString inputPath; /**< query image (file path) */
        int threads = 0;
        QString device = QStringLiteral("auto");
        /** Text prompts ("nose", "left eye", ...). */
        QStringList kpsTexts;
        /** Optional 1-shot visual prompt. */
        QString supportImagePath;
        QVector<QPointF> supportKps; /**< support-image pixel coords */
        /** Optional ROI [x1,y1,x2,y2] on the query image (single-object). */
        bool hasBbox = false;
        float bbox[4] = {0, 0, 0, 0};
        /** Multi-object mode: run YOLO-World first, then GKD per box. */
        bool multiObject = false;
        QStringList objectClasses; /**< open-vocabulary class names */
        QString yoloModelPath;     /**< YOLO-World GGUF (WOLD family) */
        float yoloConf = 0.25f;
        /** Keypoints below this score are dropped from the result. */
        float minScore = 0.30f;
        bool addResultToDb = true;
        QString savePngDir;
    };

    explicit GKDWorker(const Settings& settings, QObject* parent = nullptr);
    ~GKDWorker() override;

    /** Move the pending model contexts back to the main thread and free
     *  them. Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void taskStage(const QString& stage, int percent = -1);
    void resultReady(const GKDRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    bool detectSingleObject(QVector<GKDKeypointSet>* sets, QString* error);
    bool detectMultiObject(QVector<GKDKeypointSet>* sets, QString* error);
    bool ensureGkdContext(QString* error);
#endif

    Settings m_settings;
    aicore_cancel_token* m_cancelToken = nullptr;
    void* m_pendingGkdCtx = nullptr;  /**< aicore_gkd_ctx* (opaque) */
    void* m_pendingYoloCtx = nullptr; /**< aicore_yolo_ctx* (opaque) */
};
