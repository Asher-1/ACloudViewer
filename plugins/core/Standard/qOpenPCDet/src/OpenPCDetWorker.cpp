// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OpenPCDetWorker.h"

#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>

#include <QtCompat.h>

#include <ecvPointCloud.h>

#ifdef AICore_ENABLED
#include "aicore/openpcdet_capi.h"
#include "ecvAICoreRuntimeHelpers.h"
#endif

namespace {

// Case-insensitive intensity-ish scalar-field lookup ("intensity" from
// LAS/E57, "Intensity" from CloudCompare heritage, "reflectance" from some
// PCD/ROS exports).
bool isIntensityFieldName(const QString& name) {
    const QString lower = name.toLower();
    return lower == QStringLiteral("intensity") ||
           lower == QStringLiteral("reflectance");
}

#ifdef AICore_ENABLED
// Shared typed-result fill of the detect / detect_scene paths: detections,
// backend-resolved device and pipeline timings of the most recent call.
bool fillResult(aicore_openpcdet_ctx* ctx,
                double runtimeMs,
                OpenPCDetRunResult* resultOut,
                QString* error) {
    aicore_openpcdet_geometry geometry{};
    if (aicore_openpcdet_geometry_of(ctx, &geometry) != 0) {
        *error = QObject::tr("Cannot read the model geometry");
        return false;
    }
    OpenPCDetRunResult& result = *resultOut;
    result.numClass = geometry.num_class;
    result.runtimeMs = runtimeMs;
    // Backend-resolved device (may differ from the request when the GPU
    // lease failed and openpcdet fell back to CPU).
    const char* resolvedDevice = aicore_openpcdet_context_device(ctx);
    result.resolvedDevice = (resolvedDevice && resolvedDevice[0])
                                    ? QString::fromUtf8(resolvedDevice)
                                    : QStringLiteral("cpu");

    const int count = aicore_openpcdet_detection_count(ctx);
    result.detections.reserve(count > 0 ? count : 0);
    for (int i = 0; i < count; ++i) {
        aicore_openpcdet_detection d{};
        if (aicore_openpcdet_detection_at(ctx, i, &d) != 0) continue;
        OpenPCDetDetection out;
        out.x = d.x;
        out.y = d.y;
        out.z = d.z;
        out.dx = d.dx;
        out.dy = d.dy;
        out.dz = d.dz;
        out.heading = d.heading;
        out.score = d.score;
        out.label = d.label;
        out.className = OpenPCDetHelpers::className(d.label,
                                                    geometry.num_class);
        result.detections.append(out);
    }

    aicore_pipeline_timings t{};
    if (aicore_openpcdet_last_pipeline_timings(ctx, &t) == 0) {
        result.timingValidFields = t.valid_fields;
        result.preprocessMs = t.preprocess_ms;
        result.inferenceMs = t.inference_ms;
        result.postprocessMs = t.postprocess_ms;
        result.e2eMs = t.e2e_ms;
    }
    return true;
}
#endif  // AICore_ENABLED

}  // namespace

OpenPCDetWorker::OpenPCDetWorker(const Settings& settings,
                                 ccPointCloud* cloud,
                                 QObject* parent)
    : QThread(parent), m_settings(settings), m_cloud(cloud) {}

OpenPCDetWorker::~OpenPCDetWorker() {
    // onTaskFinished deleteLater()s the worker from a queued slot while
    // run() may still be returning; ~QThread aborts on a running thread
    // and the pending context below must outlive every run() access.
    if (isRunning()) {
        requestTaskCancel();
        wait();
    }
    releaseContextOnMainThread();
}

void OpenPCDetWorker::releaseContextOnMainThread() {
    // The context is created on the worker thread; destroy it here (main
    // thread) so GPU teardown never races the render thread.
#ifdef AICore_ENABLED
    ecvAICoreRuntime::releasePending(m_pendingCtx, &aicore_openpcdet_free);
#endif
}

void OpenPCDetWorker::requestTaskCancel() {
    qtCompatStoreRelaxed(m_cancelRequested, 1);
}

void OpenPCDetWorker::run() {
#ifdef AICore_ENABLED
    const bool ok = runInference();
    emit taskFinished(ok);
#else
    emit logMessage(tr("[OpenPCDet] AICore is not enabled in this build."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED

bool OpenPCDetWorker::runInference() {
    auto taskGuard = ecvAICoreRuntime::makeDeviceTaskLock(m_settings.device);
    if (!taskGuard.isLocked()) {
        emit logMessage(
                tr("[OpenPCDet] Failed to acquire the inference device; "
                   "another task is running."));
        return false;
    }
    if (qtCompatLoadRelaxed(m_cancelRequested) != 0) {
        emit logMessage(tr("[OpenPCDet] Cancelled before inference."));
        return false;
    }

    aicore_openpcdet_options* opts = aicore_openpcdet_options_new();
    if (!opts) {
        emit logMessage(tr("[OpenPCDet] Failed to allocate options."));
        return false;
    }
    aicore_openpcdet_options_set_device(opts,
                                        m_settings.device.toUtf8().constData());
    aicore_openpcdet_options_set_threads(opts, m_settings.threads);
    if (m_settings.scoreThresh > 0.0f) {
        aicore_openpcdet_options_set_score_thresh(opts,
                                                  m_settings.scoreThresh);
        emit logMessage(tr("[OpenPCDet] Score threshold override: %1")
                                .arg(m_settings.scoreThresh, 0, 'f', 2));
    }
    if (m_settings.nmsThresh > 0.0f) {
        aicore_openpcdet_options_set_nms_thresh(opts, m_settings.nmsThresh);
        emit logMessage(tr("[OpenPCDet] NMS IoU override: %1")
                                .arg(m_settings.nmsThresh, 0, 'f', 2));
    }
    if (m_settings.maxBoxes > 0) {
        aicore_openpcdet_options_set_max_boxes(opts, m_settings.maxBoxes);
    }

    emit logMessage(tr("[OpenPCDet] Loading model: %1 (device=%2, threads=%3)")
                            .arg(QFileInfo(m_settings.modelPath).fileName(),
                                 m_settings.device)
                            .arg(m_settings.threads));
    emit progressUpdate(0, 4);

    m_pendingCtx = aicore_openpcdet_load_opts(
            m_settings.modelPath.toUtf8().constData(), opts);
    aicore_openpcdet_options_free(opts);
    if (!m_pendingCtx || !aicore_openpcdet_is_ready(m_pendingCtx)) {
        const char* err = m_pendingCtx ? aicore_openpcdet_last_error(m_pendingCtx)
                                       : "context allocation failed";
        emit logMessage(tr("[OpenPCDet] Model load failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }
    if (qtCompatLoadRelaxed(m_cancelRequested) != 0) {
        emit logMessage(tr("[OpenPCDet] Cancelled after model load."));
        return false;
    }

    aicore_openpcdet_geometry geometry{};
    if (aicore_openpcdet_geometry_of(m_pendingCtx, &geometry) != 0) {
        emit logMessage(tr("[OpenPCDet] Cannot read the model geometry."));
        return false;
    }
    emit logMessage(
            tr("[OpenPCDet] Model ready: arch=%1, classes=%2, "
               "point features=%3, grid=%4x%5x%6, voxel=(%7, %8, %9)")
                    .arg(QString::fromUtf8(geometry.arch))
                    .arg(geometry.num_class)
                    .arg(geometry.num_point_features)
                    .arg(geometry.grid_size_x)
                    .arg(geometry.grid_size_y)
                    .arg(geometry.grid_size_z)
                    .arg(geometry.voxel_size_x, 0, 'f', 2)
                    .arg(geometry.voxel_size_y, 0, 'f', 2)
                    .arg(geometry.voxel_size_z, 0, 'f', 2));
    emit progressUpdate(1, 4);

    // ---- Input routing: scene dir / point file / active cloud -------------
    // The multi-modal checkpoints (CaDDN / BEVFusion) reject a points-only
    // detect call, so a scene directory is mandatory for them; LiDAR-only
    // checkpoints accept a scene directory too (the engine's base
    // forward_scene falls back to the point branch) but never require one.
    const bool sceneRun = !m_settings.sceneDir.isEmpty();
    const QString arch = QString::fromUtf8(geometry.arch);
    const bool modelNeedsScene = arch == QLatin1String("openpcdet.caddn") ||
                                 arch == QLatin1String("openpcdet.bevfusion");
    if (modelNeedsScene && !sceneRun) {
        emit logMessage(
                tr("[OpenPCDet] %1 is a multi-modal checkpoint — select a "
                   "scene directory (images + calibration) first.")
                        .arg(QFileInfo(m_settings.modelPath).fileName()));
        return false;
    }

    std::vector<float> points;
    int nPoints = 0;
    QString error;
    bool havePoints = false;
    if (!m_settings.pointsFile.isEmpty()) {
        if (!loadPointsFile(m_settings.pointsFile, geometry.num_point_features,
                            &points, &nPoints, &error)) {
            emit logMessage(tr("[OpenPCDet] %1").arg(error));
            return false;
        }
        havePoints = true;
        emit logMessage(tr("[OpenPCDet] Loaded %1 point(s) from %2")
                                .arg(nPoints)
                                .arg(QFileInfo(m_settings.pointsFile)
                                             .fileName()));
    } else if (m_cloud && m_cloud->size() > 0) {
        if (geometry.num_point_features >= 4 &&
            !cloudHasIntensityField(m_cloud)) {
            emit logMessage(tr("[OpenPCDet] No intensity scalar field found — "
                               "the 4th point feature is filled with 0."));
        }
        if (geometry.num_point_features == 5) {
            // nuScenes checkpoints expect [x, y, z, intensity, ring/velocity];
            // the ring/velocity channel is not carried by ccPointCloud scalar
            // fields, so it is zero-filled.
            emit logMessage(tr("[OpenPCDet] 5-feature (nuScenes) model: the "
                               "5th point feature is filled with 0."));
        }
        if (!extractCloudPoints(m_cloud, geometry.num_point_features, &points,
                                &nPoints, &error)) {
            emit logMessage(tr("[OpenPCDet] %1").arg(error));
            return false;
        }
        havePoints = true;
    }
    if (!havePoints && !sceneRun) {
        emit logMessage(tr("[OpenPCDet] Select a non-empty point cloud or a "
                           "point file first."));
        return false;
    }
    if (qtCompatLoadRelaxed(m_cancelRequested) != 0) {
        emit logMessage(tr("[OpenPCDet] Cancelled after point extraction."));
        return false;
    }
    emit progressUpdate(2, 4);

    OpenPCDetRunResult result;
    result.modelPath = m_settings.modelPath;
    result.sourceLabel =
            sceneRun
                    ? QFileInfo(m_settings.sceneDir).fileName()
                    : (!m_settings.pointsFile.isEmpty()
                               ? QFileInfo(m_settings.pointsFile)
                                         .completeBaseName()
                               : (m_cloud ? m_cloud->getName()
                                          : QStringLiteral("input")));
    if (sceneRun) {
        // Owned scene buffers come from aicore_openpcdet_load_scene_dir; the
        // point buffer (when available) stays caller-owned.
        aicore_openpcdet_scene scene{};
        char* sceneError = nullptr;
        if (aicore_openpcdet_load_scene_dir(
                    m_settings.sceneDir.toUtf8().constData(), &scene,
                    &sceneError) != 0) {
            emit logMessage(tr("[OpenPCDet] Scene load failed: %1")
                                    .arg(sceneError ? QString::fromUtf8(
                                                            sceneError)
                                                    : tr("unknown error")));
            aicore_openpcdet_free_buffer(sceneError);
            return false;
        }
        if (havePoints) {
            scene.points = points.data();
            scene.n_points = static_cast<int32_t>(nPoints);
            scene.point_dim = static_cast<int32_t>(
                    geometry.num_point_features);
        }
        emit logMessage(
                tr("[OpenPCDet] Scene ready: %1 camera(s), %2x%3x%4")
                        .arg(scene.num_cameras)
                        .arg(scene.image_height)
                        .arg(scene.image_width)
                        .arg(scene.image_channels));
        if (qtCompatLoadRelaxed(m_cancelRequested) != 0) {
            aicore_openpcdet_free_scene(&scene);
            emit logMessage(tr("[OpenPCDet] Cancelled before inference."));
            return false;
        }
        emit progressUpdate(3, 4);
        const bool ok = detectSceneBuffer(m_pendingCtx, &scene, &result,
                                          &error);
        result.sceneDir = m_settings.sceneDir;
        aicore_openpcdet_free_scene(&scene);
        if (!ok) {
            emit logMessage(tr("[OpenPCDet] %1").arg(error));
            return false;
        }
    } else {
        emit progressUpdate(3, 4);
        if (!detectBuffer(m_pendingCtx, points.data(), nPoints,
                          geometry.num_point_features, &result, &error)) {
            emit logMessage(tr("[OpenPCDet] %1").arg(error));
            return false;
        }
    }
    emit logMessage(
            tr("[OpenPCDet] %1 box(es) from %2 point(s) in %3 ms "
               "(device=%4)")
                    .arg(result.detections.size())
                    .arg(result.numPoints)
                    .arg(result.runtimeMs, 0, 'f', 1)
                    .arg(result.resolvedDevice));
    emit progressUpdate(4, 4);
    emit resultReady(result);
    return true;
}

bool OpenPCDetWorker::detectBuffer(aicore_openpcdet_ctx* ctx,
                                   const float* points,
                                   int nPoints,
                                   int pointDim,
                                   OpenPCDetRunResult* resultOut,
                                   QString* error) {
    if (ctx == nullptr || !aicore_openpcdet_is_ready(ctx)) {
        *error = tr("Model context is not ready");
        return false;
    }
    aicore_openpcdet_geometry geometry{};
    if (aicore_openpcdet_geometry_of(ctx, &geometry) != 0) {
        *error = tr("Cannot read the model geometry");
        return false;
    }
    if (pointDim != geometry.num_point_features) {
        *error = tr("Point feature mismatch: model expects %1, got %2")
                         .arg(geometry.num_point_features)
                         .arg(pointDim);
        return false;
    }
    if (nPoints <= 0 || points == nullptr) {
        *error = tr("Empty point buffer");
        return false;
    }

    QElapsedTimer timer;
    timer.start();
    const int detectRc = aicore_openpcdet_detect_points(
            ctx, points, static_cast<int32_t>(nPoints),
            static_cast<int32_t>(pointDim));
    const double ms = static_cast<double>(timer.elapsed());
    if (detectRc < 0) {
        const char* err = aicore_openpcdet_last_error(ctx);
        *error = tr("Inference failed: %1")
                         .arg(err ? QString::fromUtf8(err)
                                  : tr("unknown error"));
        return false;
    }

    OpenPCDetRunResult& result = *resultOut;
    result.numPoints = nPoints;
    result.pointDim = pointDim;
    return fillResult(ctx, ms, resultOut, error);
}

bool OpenPCDetWorker::detectSceneBuffer(aicore_openpcdet_ctx* ctx,
                                        const aicore_openpcdet_scene* scene,
                                        OpenPCDetRunResult* resultOut,
                                        QString* error) {
    if (ctx == nullptr || !aicore_openpcdet_is_ready(ctx)) {
        *error = tr("Model context is not ready");
        return false;
    }
    if (scene == nullptr) {
        *error = tr("Empty scene input");
        return false;
    }

    QElapsedTimer timer;
    timer.start();
    const int detectRc = aicore_openpcdet_detect_scene(ctx, scene);
    const double ms = static_cast<double>(timer.elapsed());
    if (detectRc < 0) {
        const char* err = aicore_openpcdet_last_error(ctx);
        *error = tr("Scene inference failed: %1")
                         .arg(err ? QString::fromUtf8(err)
                                  : tr("unknown error"));
        return false;
    }

    OpenPCDetRunResult& result = *resultOut;
    result.fromScene = true;
    result.numCameras = scene->num_cameras;
    result.numPoints = scene->n_points;
    result.pointDim = scene->point_dim;
    return fillResult(ctx, ms, resultOut, error);
}

bool OpenPCDetWorker::loadPointsFile(const QString& path,
                                     int pointDim,
                                     std::vector<float>* outPoints,
                                     int* outCount,
                                     QString* error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        *error = tr("Cannot open point file: %1").arg(path);
        return false;
    }
    const QString suffix = QFileInfo(path).suffix().toLower();
    if (suffix == QLatin1String("bin")) {
        return OpenPCDetHelpers::parsePointsBin(file.readAll(), pointDim,
                                                outPoints, outCount, error);
    }
    return OpenPCDetHelpers::parsePointsTxt(QString::fromUtf8(file.readAll()),
                                            pointDim, outPoints, outCount,
                                            error);
}

#endif  // AICore_ENABLED

bool OpenPCDetWorker::cloudHasIntensityField(ccPointCloud* cloud) {
    if (cloud == nullptr) return false;
    const unsigned count = cloud->getNumberOfScalarFields();
    for (unsigned i = 0; i < count; ++i) {
        const char* name = cloud->getScalarFieldName(static_cast<int>(i));
        if (name && isIntensityFieldName(QString::fromUtf8(name))) {
            return true;
        }
    }
    return false;
}

bool OpenPCDetWorker::extractCloudPoints(ccPointCloud* cloud,
                                         int expectedDim,
                                         std::vector<float>* outPoints,
                                         int* outCount,
                                         QString* error) {
    auto fail = [error](const QString& msg) {
        if (error) *error = msg;
        return false;
    };
    if (cloud == nullptr || cloud->size() == 0) {
        return fail(tr("The point cloud is empty"));
    }
    if (expectedDim < 3) {
        return fail(tr("Model expects %1 point features (minimum is 3)")
                            .arg(expectedDim));
    }

    // Locate the intensity scalar field (case-insensitive); absent or
    // mismatched-length fields fill the 4th feature with 0.
    cloudViewer::ScalarField* intensity = nullptr;
    const unsigned sfCount = cloud->getNumberOfScalarFields();
    for (unsigned i = 0; i < sfCount; ++i) {
        const char* name = cloud->getScalarFieldName(static_cast<int>(i));
        if (name && isIntensityFieldName(QString::fromUtf8(name))) {
            cloudViewer::ScalarField* sf =
                    cloud->getScalarField(static_cast<int>(i));
            if (sf && static_cast<int>(sf->size()) >=
                              static_cast<int>(cloud->size())) {
                intensity = sf;
            }
            break;
        }
    }

    const unsigned n = cloud->size();
    outPoints->assign(static_cast<size_t>(n) *
                              static_cast<size_t>(expectedDim),
                      0.0f);
    for (unsigned i = 0; i < n; ++i) {
        const CCVector3* p = cloud->getPoint(i);
        float* row = outPoints->data() + static_cast<size_t>(i) * expectedDim;
        row[0] = static_cast<float>(p->x);
        row[1] = static_cast<float>(p->y);
        row[2] = static_cast<float>(p->z);
        if (expectedDim >= 4 && intensity) {
            row[3] = static_cast<float>(intensity->getValue(i));
        }
        // Features beyond index 3 stay zero-filled (see run()'s nuScenes
        // hint).
    }
    *outCount = static_cast<int>(n);
    return true;
}
