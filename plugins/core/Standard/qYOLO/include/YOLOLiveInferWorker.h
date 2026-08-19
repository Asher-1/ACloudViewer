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
#include <QtGlobal>
#include <cstdint>

#include "YOLOModelCatalog.h"

struct aicore_yolo_ctx;

/** Serialized live YOLO inference with a context reused across frames. The
 *  model decides the path per load (detect or metric depth); Result::task
 *  tells the widget which payload is valid. */
class YOLOLiveInferWorker : public QObject {
    Q_OBJECT

public:
    struct Job {
        QImage rgb;
        QString modelPath;
        QString device;
        int threads = 0;
        float confThres = 0.25f;
        float iouThres = 0.7f;
        uint32_t topK = 300;
        quint64 generation = 0;
    };

    struct Result {
        YOLORunResult detect;   // valid when task == "detect"
        YOLODepthResult depth;  // valid when task == "depth"
        QString task;
        QString error;
        bool ok = false;
        quint64 generation = 0;
    };

    explicit YOLOLiveInferWorker(QObject* parent = nullptr);
    ~YOLOLiveInferWorker() override;

public slots:
    void runJob(YOLOLiveInferWorker::Job job);
    void releaseModel();

signals:
    void inferComplete(YOLOLiveInferWorker::Result result);

private:
    void runJobImpl(YOLOLiveInferWorker::Job job);
#ifdef AICore_ENABLED
    bool ensureModel(const Job& job, QString* error);

    aicore_yolo_ctx* m_ctx = nullptr;
    QString m_loadedModelPath;
    QString m_loadedDevice;
    int m_loadedThreads = 0;
    /* Task of the loaded context ("detect" | "depth"), cached at load time
     * so each Result knows which payload it carries. */
    QString m_loadedTask;
    /* Backend-RESOLVED device of the loaded context ("CUDA0", "cpu", ...);
     * differs from m_loadedDevice when the GPU lease failed. */
    QString m_resolvedDevice;
#endif
};

Q_DECLARE_METATYPE(YOLOLiveInferWorker::Job)
Q_DECLARE_METATYPE(YOLOLiveInferWorker::Result)
