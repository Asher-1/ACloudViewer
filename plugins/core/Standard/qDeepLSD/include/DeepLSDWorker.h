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
#include <vector>

struct aicore_cancel_token;

struct DeepLSDLineSegment {
    float x1 = 0.0f;
    float y1 = 0.0f;
    float x2 = 0.0f;
    float y2 = 0.0f;
    float score = 0.0f;
};

struct DeepLSDRunResult {
    QString imagePath;
    QString imageName;
    int width = 0;
    int height = 0;
    int originalWidth = 0;
    int originalHeight = 0;
    QImage lineVisualization;
    QImage distanceOverlay;
    std::vector<DeepLSDLineSegment> segments;
    double runtimeMs = 0.0;
    QString resolvedDevice;
};

Q_DECLARE_METATYPE(DeepLSDRunResult)

/** RGB color used for line visualization and exported LineSet objects. */
struct DeepLSDLineStyle {
    static constexpr int kRed = 0;
    static constexpr int kGreen = 255;
    static constexpr int kBlue = 0;
};

class DeepLSDWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString inputPath;
        int threads = 0;
        QString device = "auto";
        float minSegmentScore = 0.0f;
        bool computeDistanceOverlay = false;
    };

    explicit DeepLSDWorker(const Settings& settings, QObject* parent = nullptr);
    ~DeepLSDWorker() override;
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const DeepLSDRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runExtract();
#endif

    Settings m_settings;
    aicore_cancel_token* m_cancelToken = nullptr;
    struct aicore_deeplsd_ctx* m_pendingCtx = nullptr;
};
