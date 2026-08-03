// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include "FaceDetectWorker.h"
#include "FaceRegistryStore.h"

#include <QImage>
#include <QObject>
#include <QString>
#include <atomic>

/** Background AICore inference for live camera / video (serialized via inference lock). */
class FaceLiveDetectInferWorker : public QObject {
    Q_OBJECT

public:
    enum class StreamMode { Detect = 0, Recognize = 1 };

    struct Job {
        QImage inferRgb;
        QImage displayRgb;
        float inferScale = 1.f;
        QString modelPath;
        QString device;
        int threads = 0;
        float minDetectionScore = 0.5f;
        float matchThreshold = 0.52f;
        StreamMode streamMode = StreamMode::Detect;
        FaceRegistryStore* registry = nullptr;
    };

    struct Result {
        FaceDetectRunResult snapshot;
        QImage displayImage;
        int identifiedCount = 0;
        bool ok = false;
    };

    explicit FaceLiveDetectInferWorker(QObject* parent = nullptr);

public slots:
    void runJob(FaceLiveDetectInferWorker::Job job);
    void releaseModel();

signals:
    void inferComplete(FaceLiveDetectInferWorker::Result result);

private:
#ifdef AICore_ENABLED
    bool ensureModel(const Job& job);
    bool runDetectJob(const Job& job, Result* out);
    bool runRecognizeJob(const Job& job, Result* out);

    aicore_facedetect_ctx* m_ctx = nullptr;
    QString m_loadedModelPath;
#endif
};

Q_DECLARE_METATYPE(FaceLiveDetectInferWorker::Job)
Q_DECLARE_METATYPE(FaceLiveDetectInferWorker::Result)
