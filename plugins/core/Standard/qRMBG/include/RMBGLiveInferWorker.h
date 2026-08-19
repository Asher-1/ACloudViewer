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

#include "RMBGModelCatalog.h"

struct aicore_rmbg_ctx;

/** Serialized live RMBG inference with a context reused across frames. */
class RMBGLiveInferWorker : public QObject {
    Q_OBJECT

public:
    struct Job {
        QImage rgb;
        QString modelPath;
        QString device;
        int threads = 0;
        float alphaThreshold = 0.5f;
        quint64 generation = 0;
    };

    struct Result {
        RMBGRunResult snapshot;
        QString error;
        bool ok = false;
        quint64 generation = 0;
    };

    explicit RMBGLiveInferWorker(QObject* parent = nullptr);
    ~RMBGLiveInferWorker() override;

public slots:
    void runJob(RMBGLiveInferWorker::Job job);
    void releaseModel();

signals:
    void inferComplete(RMBGLiveInferWorker::Result result);

private:
    void runJobImpl(RMBGLiveInferWorker::Job job);
#ifdef AICore_ENABLED
    bool ensureModel(const Job& job, QString* error);

    aicore_rmbg_ctx* m_ctx = nullptr;
    QString m_loadedModelPath;
    QString m_loadedDevice;
    int m_loadedThreads = 0;
    // Model metadata parsed once per loaded context (see ensureModel).
    RMBGRunResult m_info;
#endif
};

Q_DECLARE_METATYPE(RMBGLiveInferWorker::Job)
Q_DECLARE_METATYPE(RMBGLiveInferWorker::Result)
