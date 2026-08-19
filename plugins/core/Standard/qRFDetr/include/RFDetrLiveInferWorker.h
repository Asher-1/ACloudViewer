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

#include "RFDetrModelCatalog.h"

struct aicore_rfdetr_ctx;

/** Serialized live RF-DETR inference with a context reused across frames. */
class RFDetrLiveInferWorker : public QObject {
    Q_OBJECT

public:
    struct Job {
        QImage rgb;
        QString modelPath;
        QString device;
        int threads = 0;
        float threshold = 0.5f;
        uint32_t topK = 300;
        quint64 generation = 0;
    };

    struct Result {
        RFDetrRunResult snapshot;
        QString error;
        bool ok = false;
        quint64 generation = 0;
    };

    explicit RFDetrLiveInferWorker(QObject* parent = nullptr);
    ~RFDetrLiveInferWorker() override;

public slots:
    void runJob(RFDetrLiveInferWorker::Job job);
    void releaseModel();

signals:
    void inferComplete(RFDetrLiveInferWorker::Result result);

private:
    void runJobImpl(RFDetrLiveInferWorker::Job job);
#ifdef AICore_ENABLED
    bool ensureModel(const Job& job, QString* error);

    aicore_rfdetr_ctx* m_ctx = nullptr;
    QString m_loadedModelPath;
    QString m_loadedDevice;
    int m_loadedThreads = 0;
    /* Backend-RESOLVED device of the loaded context ("CUDA0", "cpu", ...);
     * differs from m_loadedDevice when the GPU lease failed. */
    QString m_resolvedDevice;
#endif
};

Q_DECLARE_METATYPE(RFDetrLiveInferWorker::Job)
Q_DECLARE_METATYPE(RFDetrLiveInferWorker::Result)
