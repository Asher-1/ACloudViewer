// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QImage>
#include <QObject>
#include <QPointF>
#include <QString>
#include <QThread>
#include <QVector3D>
#include <vector>

struct FaceDetectBox {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
    float landmarks[5][2] = {};
    int age = -1;
    char gender = '?';
    std::vector<QPointF> denseLandmarks2d;
    std::vector<QVector3D> denseLandmarks3d;
};

struct FaceDetectRunResult {
    QString imagePath;
    QString imageName;
    QString secondImagePath;
    QImage annotatedImage;
    std::vector<FaceDetectBox> faces;
    double runtimeMs = 0.0;
    float verifyDistance = -1.0f;
    int verifyMatched = -1;
    int totalDetected = 0;
    int rejectedByScore = 0;
    float minDetectionScoreUsed = 0.5f;
    QString mode;
    QString resolvedDevice;
    /** Raw JSON from AICore (detect/analyze) or synthesized verify payload. */
    QByteArray resultJson;
};

Q_DECLARE_METATYPE(FaceDetectRunResult)

struct aicore_cancel_token;

class FaceDetectWorker : public QThread {
    Q_OBJECT

public:
    enum class Mode { Detect, Analyze, Verify, DenseLandmarks };

    struct Settings {
        QString modelPath;
        QString landmarkModelPath;
        QString inputPath;
        QString secondInputPath;
        int threads = 0;
        QString device = "auto";
        Mode mode = Mode::Detect;
        float verifyThreshold = 0.52f;
        float minDetectionScore = 0.5f;
        bool antiSpoof = false;
    };

    explicit FaceDetectWorker(const Settings& settings,
                              QObject* parent = nullptr);
    ~FaceDetectWorker() override;
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const FaceDetectRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
#endif

    Settings m_settings;
    struct aicore_facedetect_ctx* m_pendingCtx = nullptr;
    struct aicore_facedetect_ctx* m_pendingLandmarkCtx = nullptr;
    aicore_cancel_token* m_cancelToken = nullptr;
};
