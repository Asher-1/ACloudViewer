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
#include <atomic>

#include "FaceDetectWorker.h"
#include "FaceRegistryStore.h"

/** Background AICore inference for live camera / video (serialized via
 * inference lock). */
class FaceLiveDetectInferWorker : public QObject {
    Q_OBJECT

public:
    enum class StreamMode { Detect = 0, Recognize = 1 };

    struct Job {
        QImage inferRgb;
        float inferScale = 1.f;
        QString modelPath;
        QString device;
        int threads = 0;
        float minDetectionScore = 0.5f;
        float matchThreshold = 0.65f;
        StreamMode streamMode = StreamMode::Detect;
        FaceRegistryStore* registry = nullptr;
        quint64 generation = 0;
    };

    struct Result {
        FaceDetectRunResult snapshot;
        QImage displayImage;
        QVector<QString>
                labels;  // recognize-mode labels (parallel to snapshot.faces)
        int identifiedCount = 0;
        bool ok = false;
        quint64 generation = 0;
    };

    explicit FaceLiveDetectInferWorker(QObject* parent = nullptr);

public slots:
    void runJob(FaceLiveDetectInferWorker::Job job);
    void releaseModel();
    void preloadModel(const QString& modelPath,
                      const QString& device,
                      int threads);

signals:
    void inferComplete(FaceLiveDetectInferWorker::Result result);
    void modelPreloadComplete(bool ok);

private:
#ifdef AICore_ENABLED
    bool ensureModel(const Job& job);
    bool runDetectJob(const Job& job, Result* out);
    bool runRecognizeJob(const Job& job, Result* out);

    aicore_facedetect_ctx* m_ctx = nullptr;
    QString m_loadedModelPath;
    QString m_loadedDevice;
    int m_loadedThreads = 0;
#endif
};

Q_DECLARE_METATYPE(FaceLiveDetectInferWorker::Job)
Q_DECLARE_METATYPE(FaceLiveDetectInferWorker::Result)
