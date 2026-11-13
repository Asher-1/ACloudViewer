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
#include <memory>
#include <vector>

struct aicore_querysplat_ctx;

/** Result envelope of one QuerySplat run. */
struct QuerySplatResult {
    QString sourceLabel;
    QString decoderPath;
    QString encoderPath;
    QString resolvedDevice;
    double e2eMs = 0.0;
    int viewCount = 0;
    int width = 0;
    int height = 0;
    int shDegree = 0;

    // Typed gaussian buffer ownership transfer (aicore_free_buffer deleter;
    // no second full-size copy). Layout: N x stride
    // [pos(3) | opacity | scale(3) | rot(4) | color].
    int64_t gaussianCount = 0;
    int gaussianStride = 0;
    std::shared_ptr<float> gaussians;

    // Per-view colorized depth rasters (turbo) over the preprocessed grid.
    QVector<QImage> depthViews;
    // Per-view rasterized renders (RGB, gsplat semantics) of the predicted
    // gaussians from the predicted input cameras.
    QVector<QImage> renderViews;

    // Depth point cloud (XYZ + RGB888) reconstructed from the VGGT depth
    // maps through the predicted cameras.
    QVector<float> points;
    QVector<quint8> colors;
    int pointCount = 0;

    // Per-view world-from-camera 4x4 (row-major 16) + intrinsics
    // [fx, fy, cx, cy] over the preprocessed grid.
    QVector<float> camPoses16;
    QVector<float> camIntrinsics;

    // 3DGS PLY written next to the outputs (opacity-filtered).
    QString plyPath;
    // compact machine-readable run summary
    QByteArray resultJson;
    // TTO losses when enabled (l1, ssim, lpips per step)
    struct TtoStep {
        double l1;
        double ssim;
        double lpips;
    };
    QVector<TtoStep> ttoSteps;
};

Q_DECLARE_METATYPE(QuerySplatResult)

/** Background QuerySplat inference worker. The model context is created
 *  inside run() on the worker thread and released there before the thread
 *  finishes (the AICore context is not GUI-bound; the plugin holds no
 *  context across GUI thread hops). */
class QuerySplatWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString decoderPath;
        QString encoderPath;  // empty = catalog default (shared map_anything)
        QStringList imagePaths;
        int threads = 0;
        QString device = QStringLiteral("auto");
        int imageSize = 0;  // 0 = decoder GGUF metadata resolution
        bool ttoEnabled = false;
        int ttoSteps = 20;
        double ttoLr = 5e-3;
        double ttoLpipsWeight = 0.05;
        float opacityThreshold = 0.05f;
        float renderBackground = 0.0f;  // gsplat raster background (grayscale)
        bool addCameras = true;
        bool addDepthViews = true;
        bool addRenderViews = true;
        bool addDepthPointCloud = true;
        bool exportPly = true;
        QString outputDir;  // PLY destination; empty = no export
    };

    explicit QuerySplatWorker(QObject* parent = nullptr);
    ~QuerySplatWorker() override;

    void setSettings(const Settings& settings);

protected:
    void run() override;

signals:
    void progressUpdate(int current, int total);
    void stageUpdate(const QString& stage);
    void logMessage(const QString& msg);
    void resultReady(const QuerySplatResult& result);

private:
    bool loadContext();
    QImage colorizeDepth(const float* depth,
                         const float* conf,
                         int w,
                         int h,
                         int view) const;
    static QImage rgb8ToImage(const unsigned char* rgb8, int w, int h);

    Settings m_settings;
    aicore_querysplat_ctx* m_ctx = nullptr;
};
