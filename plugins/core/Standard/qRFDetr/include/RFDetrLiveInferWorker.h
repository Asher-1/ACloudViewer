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
#include <QVector>
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
        /** Class allowlist (empty = detect all classes). Passed through to
         *  the engine's post-processing; part of the ctx-reuse key so a
         *  changed filter reloads the context. */
        QVector<uint32_t> classFilter;
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
    /** Emitted once after a (re)load with the model-info JSON envelope
     *  (same payload as RFDetrWorker::modelInfoReady) so the dialog can
     *  populate its class-filter list in live mode too. */
    void modelInfoReady(const QString& info);

private:
    void runJobImpl(RFDetrLiveInferWorker::Job job);
#ifdef AICore_ENABLED
    bool ensureModel(const Job& job, QString* error);

    aicore_rfdetr_ctx* m_ctx = nullptr;
    QString m_loadedModelPath;
    QString m_loadedDevice;
    int m_loadedThreads = 0;
    /* Class allowlist the loaded context was created with (empty = all
     * classes); a change triggers a context reload in ensureModel(). */
    QVector<uint32_t> m_loadedClassFilter;
    /* Backend-RESOLVED device of the loaded context ("CUDA0", "cpu", ...);
     * differs from m_loadedDevice when the GPU lease failed. */
    QString m_resolvedDevice;
#endif
};

Q_DECLARE_METATYPE(RFDetrLiveInferWorker::Job)
Q_DECLARE_METATYPE(RFDetrLiveInferWorker::Result)
