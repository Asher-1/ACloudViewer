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
#include <cstdint>
#include <vector>

struct aicore_cancel_token;

/** One colorized depth view of a MapAnything run (typed float map kept for
 *  future exporters, QImage for the DB raster). */
struct MapAnythingViewResult {
    QImage depthImage;  // colorized (turbo) depth raster, CROPPED to the
                        // view's valid (pre-pad) rect
    QImage confImage;   // colorized normalized-confidence raster, same crop
    QImage maskImage;   // grayscale validity mask (255 = in-mask), mapanything
                        // architecture only, same crop
    QVector<float> depthMap;
    int width = 0;
    int height = 0;
    QString sourcePath;
};

/** Result envelope of one MapAnything multi-view inference. */
struct MapAnythingResult {
    QString sourceLabel;
    QString modelVariant;
    QString resolvedDevice;
    QString modelPath;
    QString architecture;
    int viewCount = 0;
    double e2eMs = 0.0;

    // Merged world-frame cloud (XYZ + RGB888), one entry per point.
    QVector<float> points;   // 3 per point
    QVector<quint8> colors;  // 3 per point
    int pointCount = 0;

    QVector<MapAnythingViewResult> views;
    QByteArray resultJson;

    // Per-view camera-TO-world 4x4 (row-major, 16 floats) + the pinhole
    // focal in pixels over the preprocessed grid. Filled only when the
    // architecture publishes intrinsics (the vggt family pose_enc).
    bool hasCameras = false;
    int camWidth = 0;
    int camHeight = 0;
    QVector<float> camPoses16;  // viewCount * 16
    QVector<float> camFocalPx;  // viewCount

    // mapanything architecture: metric scale factor the predicted local
    // geometry must be multiplied by to become metric.
    bool hasMetricScale = false;
    float metricScale = 1.0f;

    // vggt-omega-256-text variant only: L2-normalized language-aligned
    // embedding (textEmbeddingDim floats). No geometric meaning - exported
    // as a sidecar artifact, never as a DBTree entity.
    QByteArray textEmbedding;
    int textEmbeddingDim = 0;

    // Opt-in (Settings::addPerViewClouds, default off): per-view world-frame
    // clouds in the same XYZ/RGB888 layout as the merged cloud. Large.
    QVector<QVector<float>> perViewPoints;
    QVector<QVector<quint8>> perViewColors;
};

Q_DECLARE_METATYPE(MapAnythingResult)

/** Background MapAnything inference worker (multi-view). The model context
 *  is created inside run() on the worker thread; it is released on the main
 *  thread via releaseContextOnMainThread() so GPU teardown never races the
 *  render thread. */
class MapAnythingWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QStringList imagePaths;  // >= 1 view
        int threads = 0;
        QString device = QStringLiteral("auto");
        int imageSize = 0;  // 0 = checkpoint default
        QString resizeMode = QStringLiteral("balanced");
        bool addPointCloud = true;
        bool addDepthRasters = true;
        bool addPerViewClouds = false;  // large: one entity per view
        float minConf = 0.5f;  // normalized (sigmoid for pi3 raw logits)
        // Official end-to-end filter stack defaults (upstream
        // e2e_full_scene.py, verified against the torch reference):
        float confPercentile = 20.0f;  // scene-wide p20 floor, clamped at 2.0
        float depthEdgeRtol = 0.03f;   // 3x3 depth-jump edge removal
        float voxelSize = 0.02f;       // conf-priority voxel merge (0 = off)
        bool responsiveStreaming = false;  // force chunked forwards (K=16)
                                           // for progress + cancel points;
                                           // slightly below single-pass
                                           // quality (official chain mode)
        int maxFramesPerPass = 64;  // > this: automatic chunked streaming
                                    // (~118 MiB/frame f16@512, VRAM-measured
                                    //  upstream: 38 frames peak at 8.75 GiB)
    };

    explicit MapAnythingWorker(const Settings& settings,
                               QObject* parent = nullptr);
    ~MapAnythingWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    void requestTaskCancel();

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const MapAnythingResult& result);
    void taskFinished(bool success);
    void modelInfoReady(const QString& info);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
    /** Crop/colorize one view's depth, confidence and mask rasters from the
     *  live chunk result. \p i indexes the chunk tensors, \p g the global
     *  view (imagePaths). Called once per view by its first-cover chunk. */
    void appendViewRasters(struct aicore_map_anything_ctx* ctx, int i, int g);
#endif

    Settings m_settings;
    struct aicore_map_anything_ctx* m_pendingCtx = nullptr;
    aicore_cancel_token* m_cancelToken = nullptr;
    MapAnythingResult m_result;
};
