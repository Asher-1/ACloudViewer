// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QAtomicInt>
#include <QObject>
#include <QString>
#include <QThread>
#include <vector>

#include "OpenPCDetModelCatalog.h"

class ccPointCloud;
struct aicore_openpcdet_ctx;
struct aicore_openpcdet_scene;

/** Background OpenPCDet inference worker (one selected ccPointCloud per
 *  run). The model context is created inside run() on the worker thread;
 *  it is released on the main thread via releaseContextOnMainThread() so
 *  GPU teardown never races the render thread. */
class OpenPCDetWorker : public QThread {
    Q_OBJECT

public:
    struct Settings {
        QString modelPath;
        QString device = QStringLiteral("auto");
        int threads = 0;
        /** Detection score threshold override; <= 0 keeps the GGUF-declared
         *  value (NaN semantics of the C API flattened to "0 = follow
         *  model" so the value stays settings-file friendly). */
        float scoreThresh = 0.0f;
        /** Rotated NMS IoU override; <= 0 keeps the GGUF value. */
        float nmsThresh = 0.0f;
        /** Maximum returned boxes; <= 0 keeps the GGUF value. */
        int maxBoxes = 0;
        /** Optional raw point file (.bin KITTI float32 / ASCII .txt). When
         *  set it takes precedence over the ccPointCloud input; parsed with
         *  OpenPCDetHelpers::parsePointsBin / parsePointsTxt — the exact
         *  semantics of the headless batch command. */
        QString pointsFile;
        /** Optional multi-modal scene directory (aicore_openpcdet_load_
         *  scene_dir payload: images.bin + calibration + scene.json). When
         *  set the run goes through detect_scene. */
        QString sceneDir;
    };

    OpenPCDetWorker(const Settings& settings,
                    ccPointCloud* cloud,
                    QObject* parent = nullptr);
    ~OpenPCDetWorker() override;

    /** Move the pending model context back to the main thread and free it.
     *  Safe to call from the main thread while the worker is idle. */
    void releaseContextOnMainThread();
    /** Cooperative stage-boundary cancel (load / extract / detect). The
     *  engine's detect call itself has no cancellation hook — a running
     *  detection always completes before the worker observes the flag. */
    void requestTaskCancel();

#ifdef AICore_ENABLED
    /** Run one point buffer through a loaded context (detect, typed-result
     *  conversion, timings). Shared by run() and the headless
     *  -OPENPCDET_DETECT batch command so both paths produce identical
     *  results. Returns false with \p error set on failure. */
    static bool detectBuffer(aicore_openpcdet_ctx* ctx,
                             const float* points,
                             int nPoints,
                             int pointDim,
                             OpenPCDetRunResult* resultOut,
                             QString* error);

    /** Run one loaded multi-modal scene through a loaded context
     *  (detect_scene, typed-result conversion, timings). Same result
     *  access as detectBuffer; marks the result as a scene run. */
    static bool detectSceneBuffer(aicore_openpcdet_ctx* ctx,
                                  const aicore_openpcdet_scene* scene,
                                  OpenPCDetRunResult* resultOut,
                                  QString* error);

    /** Parses one raw point file (.bin KITTI float32 or whitespace-
     *  separated ASCII .txt) into a row-major [n, pointDim] buffer —
     *  shared by the dialog's file input and the batch command. */
    static bool loadPointsFile(const QString& path,
                               int pointDim,
                               std::vector<float>* outPoints,
                               int* outCount,
                               QString* error);
#endif

    /** Extracts a contiguous [n_points, expectedDim] float32 buffer from a
     *  ccPointCloud: x, y, z plus the intensity scalar field (first field
     *  named intensity/reflectance, case-insensitive; missing intensity
     *  fills the 4th feature with 0). NuScenes-style 5-feature models get
     *  the 5th feature zero-filled. Static so tests and the batch command
     *  reuse the exact GUI input path. Returns false with \p error on an
     *  empty cloud or a feature-count mismatch. */
    static bool extractCloudPoints(ccPointCloud* cloud,
                                   int expectedDim,
                                   std::vector<float>* outPoints,
                                   int* outCount,
                                   QString* error);

    /** True when the cloud carries a scalar field named intensity or
     *  reflectance (case-insensitive) — the field extractCloudPoints()
     *  picks as the 4th feature. */
    static bool cloudHasIntensityField(ccPointCloud* cloud);

signals:
    void logMessage(const QString& msg);
    void progressUpdate(int current, int total);
    void resultReady(const OpenPCDetRunResult& result);
    void taskFinished(bool success);

protected:
    void run() override;

private:
#ifdef AICore_ENABLED
    bool runInference();
#endif

    Settings m_settings;
    ccPointCloud* m_cloud = nullptr;
    struct aicore_openpcdet_ctx* m_pendingCtx = nullptr;
    QAtomicInt m_cancelRequested;
};
