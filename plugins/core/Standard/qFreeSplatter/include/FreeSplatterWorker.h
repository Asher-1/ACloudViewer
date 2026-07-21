// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QObject>
#include <QString>
#include <QThread>
#include <QVector>

// Result of a FreeSplatter inference run
struct FreeSplatterResult {
    QString sourceName;
    int nViews = 0;
    int height = 0;
    int width = 0;
    int gaussianChannels = 0;
    int shDegree = 1;
    QVector<float> gaussians;  // nViews * H * W * gc float32
    bool hasPoses = false;
    QVector<float> cam2world;  // nViews * 16 float32 (row-major 4x4)
    float focal = 0.0f;
};

Q_DECLARE_METATYPE(FreeSplatterResult)

struct aicore_gaussian_ctx;

class FreeSplatterWorker : public QThread {
    Q_OBJECT

public:
    enum class Mode { Reconstruct, ModelInfo };

    struct Settings {
        Mode mode = Mode::Reconstruct;
        QString modelPath;
        QStringList inputPaths;
        int threads = 0;
        QString device = "auto";  // auto | cpu | cuda | vulkan | opencl
        float opacityThreshold = 0.05f;
        bool estimatePoses = false;
    };

    explicit FreeSplatterWorker(const Settings& settings,
                                QObject* parent = nullptr);

    void run() override;

    // Must be called on the main/GUI thread (ggml/CUDA teardown is not safe
    // from the worker thread after inference in ACloudViewer's mixed GPU
    // process).
    void releaseContextOnMainThread();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const FreeSplatterResult& result);
    void modelInfoReady(const QString& info);
    void taskFinished(bool success);

private:
    aicore_gaussian_ctx* loadModel();
    void stashContext(aicore_gaussian_ctx* ctx);
    bool runReconstruct();
    bool runModelInfo();

    Settings m_settings;
    aicore_gaussian_ctx* m_pendingCtx = nullptr;
};
