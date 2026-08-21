// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOWorker.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QImage>

#ifdef AICore_ENABLED
#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"
#endif

YOLOWorker::YOLOWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
}

YOLOWorker::~YOLOWorker() {
    releaseContextOnMainThread();
#ifdef AICore_ENABLED
    if (m_cancelToken) {
        aicore_cancel_token_free(m_cancelToken);
        m_cancelToken = nullptr;
    }
#endif
}

void YOLOWorker::releaseContextOnMainThread() {
    // The context is created on the worker thread; destroy it here (main
    // thread) so GPU teardown never races the render thread.
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_yolo_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

void YOLOWorker::requestTaskCancel() {
#ifdef AICore_ENABLED
    if (m_cancelToken) aicore_cancel_token_request(m_cancelToken);
#endif
}

void YOLOWorker::run() {
#ifdef AICore_ENABLED
    const bool ok = runInference();
    emit taskFinished(ok);
#else
    emit logMessage(tr("[YOLO] AICore is not enabled in this build."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED
bool YOLOWorker::runInference() {
    // Warm up the backend on the UI thread is the caller's job; here we just
    // create the model context and run.
    aicore_yolo_options* opts = aicore_yolo_options_new();
    if (!opts) {
        emit logMessage(tr("[YOLO] Failed to allocate options."));
        return false;
    }
    aicore_yolo_options_set_device(opts,
                                   m_settings.device.toUtf8().constData());
    aicore_yolo_options_set_threads(opts, m_settings.threads);
    aicore_yolo_options_set_conf_thres(opts, m_settings.confThres);
    aicore_yolo_options_set_iou_thres(opts, m_settings.iouThres);
    aicore_yolo_options_set_top_k(opts, m_settings.topK);

    emit logMessage(tr("[YOLO] Loading model: %1 (device=%2, threads=%3)")
                            .arg(QFileInfo(m_settings.modelPath).fileName(),
                                 m_settings.device)
                            .arg(m_settings.threads));
    emit progressUpdate(0, 1);

    m_pendingCtx = aicore_yolo_load_opts(
            m_settings.modelPath.toUtf8().constData(), opts);
    aicore_yolo_options_free(opts);
    if (!m_pendingCtx || !aicore_yolo_is_ready(m_pendingCtx)) {
        const char* err = m_pendingCtx ? aicore_yolo_last_error(m_pendingCtx)
                                       : "context allocation failed";
        emit logMessage(tr("[YOLO] Model load failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    const QString modelName =
            QString::fromUtf8(aicore_yolo_context_model_name(m_pendingCtx));
    const QString task =
            QString::fromUtf8(aicore_yolo_context_task(m_pendingCtx));
    emit logMessage(
            tr("[YOLO] Model loaded: %1 (task=%2, imgsz=%3, classes=%4, "
               "end2end=%5)")
                    .arg(modelName, task)
                    .arg(aicore_yolo_context_image_size(m_pendingCtx))
                    .arg(aicore_yolo_context_num_classes(m_pendingCtx))
                    .arg(aicore_yolo_context_end2end(m_pendingCtx)));
    {
        const QString info =
                QStringLiteral(
                        "{\"model\":\"%1\",\"task\":\"%2\","
                        "\"image_size\":%3,\"num_classes\":%4,"
                        "\"end2end\":%5}")
                        .arg(modelName, task)
                        .arg(aicore_yolo_context_image_size(m_pendingCtx))
                        .arg(aicore_yolo_context_num_classes(m_pendingCtx))
                        .arg(aicore_yolo_context_end2end(m_pendingCtx));
        emit modelInfoReady(info);
    }
    emit progressUpdate(1, 1);

    if (aicore_cancel_token_requested(m_cancelToken)) {
        emit logMessage(tr("[YOLO] Cancelled before inference."));
        return false;
    }

    // Single-image inference.
    const QImage input(m_settings.inputPath);
    if (input.isNull()) {
        emit logMessage(tr("[YOLO] Failed to load image: %1")
                                .arg(m_settings.inputPath));
        return false;
    }
    const QImage rgb = input.convertToFormat(QImage::Format_RGB888);
    emit progressUpdate(0, 1);

    QByteArray packedRgb;
    const uchar* rgbData = YOLOHelpers::packedRgb888Data(rgb, &packedRgb);
    if (!rgbData) {
        emit logMessage(tr("[YOLO] Failed to pack the RGB input."));
        return false;
    }

    // The loaded model decides the path: a detect GGUF yields boxes, a
    // segment GGUF yields boxes + instance masks, a depth GGUF yields a
    // metric depth map — there is no user-side task switch that could
    // disagree with the model.
    const bool ok = (task == QStringLiteral("depth")) ? runDepth(rgb, rgbData)
                    : (task == QStringLiteral("segment"))
                            ? runSegment(rgb, rgbData)
                            : runDetect(rgb, rgbData);
    emit progressUpdate(1, 1);
    return ok;
}

bool YOLOWorker::runDetect(const QImage& rgb, const uchar* rgbData) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    char* json = aicore_yolo_detect_rgb_json(m_pendingCtx, rgbData, rgb.width(),
                                             rgb.height());
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (!json) {
        const char* err = aicore_yolo_last_error(m_pendingCtx);
        emit logMessage(tr("[YOLO] Inference failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    YOLORunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.modelPath = m_settings.modelPath;
    result.task = QStringLiteral("detect");
    result.runtimeMs = ms;
    // Backend-resolved device (may differ from the request when the GPU
    // lease failed and yolo fell back to CPU).
    const char* resolvedDevice = aicore_yolo_context_device(m_pendingCtx);
    result.resolvedDevice = (resolvedDevice && resolvedDevice[0])
                                    ? QString::fromUtf8(resolvedDevice)
                                    : m_settings.device;
    if (!YOLOHelpers::parseDetectionsJson(QByteArray(json), &result)) {
        aicore_yolo_free_buffer(json);
        emit logMessage(tr("[YOLO] Failed to parse detection JSON."));
        return false;
    }
    aicore_yolo_free_buffer(json);

    // Annotated image (boxes + labels) for DB export.
    QImage annotated = rgb;
    YOLOHelpers::drawDetections(&annotated, result.detections);
    result.annotatedImage = annotated;

    emit logMessage(
            tr("[YOLO] %1 object(s) in %2 ms (model=%3, conf=%4, iou=%5)")
                    .arg(result.detections.size())
                    .arg(ms, 0, 'f', 1)
                    .arg(result.modelVariant)
                    .arg(m_settings.confThres, 0, 'f', 2)
                    .arg(m_settings.iouThres, 0, 'f', 2));
    emit resultReady(result);
    return true;
}

bool YOLOWorker::runSegment(const QImage& rgb, const uchar* rgbData) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    aicore_yolo_segment_result* seg = aicore_yolo_seg_rgb(
            m_pendingCtx, rgbData, rgb.width(), rgb.height());
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (!seg) {
        const char* err = aicore_yolo_last_error(m_pendingCtx);
        emit logMessage(tr("[YOLO] Segmentation failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    YOLORunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.modelPath = m_settings.modelPath;
    result.task = QStringLiteral("segment");
    result.runtimeMs = ms;
    const char* resolvedDevice = aicore_yolo_context_device(m_pendingCtx);
    result.resolvedDevice = (resolvedDevice && resolvedDevice[0])
                                    ? QString::fromUtf8(resolvedDevice)
                                    : m_settings.device;
    result.modelVariant =
            QString::fromUtf8(aicore_yolo_context_model_name(m_pendingCtx));
    result.imageSize =
            static_cast<int>(aicore_yolo_context_image_size(m_pendingCtx));
    result.numClasses =
            static_cast<int>(aicore_yolo_context_num_classes(m_pendingCtx));
    result.end2end = aicore_yolo_context_end2end(m_pendingCtx) != 0;

    // Typed segment result: detections + per-instance masks.
    const int n = aicore_yolo_seg_det_count(seg);
    result.detections.reserve(n > 0 ? n : 0);
    result.masks.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_detection det = aicore_yolo_seg_det_at(seg, i);
        YOLODetection d;
        d.classId = det.class_id;
        d.x1 = det.x1;
        d.y1 = det.y1;
        d.x2 = det.x2;
        d.y2 = det.y2;
        d.score = det.score;
        result.detections.append(d);

        const aicore_yolo_plane_view view = aicore_yolo_seg_mask_at(seg, i);
        if (view.data != nullptr && view.width > 0 && view.height > 0) {
            YOLOSegMask mask;
            mask.w = view.width;
            mask.h = view.height;
            // Deep copy: the segment result is freed below and QByteArray
            // (const char*, int) allocates owned storage.
            mask.bits = QByteArray(
                    static_cast<const char*>(view.data),
                    static_cast<int>(view.row_stride_bytes) * view.height);
            result.masks.append(mask);
        }
    }
    // The typed API exposes the model's class table; fall back to the
    // deterministic palette label only when the model declares no names.
    for (int i = 0; i < result.detections.size(); ++i) {
        const char* name = aicore_yolo_seg_det_class_name(seg, i);
        result.detections[i].className =
                (name != nullptr && name[0] != '\0')
                        ? QString::fromUtf8(name)
                        : QStringLiteral("class %1")
                                  .arg(result.detections[i].classId);
    }
    result.totalDetected = n;
    aicore_yolo_seg_result_free(seg);

    // Annotated image: translucent per-class mask tint + boxes/labels.
    QImage annotated = rgb;
    YOLOHelpers::drawSegmentation(&annotated, result.masks, result.detections);
    result.annotatedImage = annotated;

    emit logMessage(
            tr("[YOLO] %1 segment(s) in %2 ms (model=%3, conf=%4, iou=%5)")
                    .arg(n)
                    .arg(ms, 0, 'f', 1)
                    .arg(result.modelVariant)
                    .arg(m_settings.confThres, 0, 'f', 2)
                    .arg(m_settings.iouThres, 0, 'f', 2));
    emit resultReady(result);
    return true;
}

bool YOLOWorker::runDepth(const QImage& rgb, const uchar* rgbData) {
    QElapsedTimer timer;
    timer.start();
    int32_t depthW = 0, depthH = 0;
    aicore_cancel_scope_begin(m_cancelToken);
    float* depth = aicore_yolo_depth_rgb(m_pendingCtx, rgbData, rgb.width(),
                                         rgb.height(), &depthW, &depthH);
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (!depth || depthW <= 0 || depthH <= 0) {
        if (depth) aicore_yolo_free_buffer(depth);
        const char* err = aicore_yolo_last_error(m_pendingCtx);
        emit logMessage(tr("[YOLO] Depth inference failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    YOLODepthResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.modelPath = m_settings.modelPath;
    result.runtimeMs = ms;
    result.width = depthW;
    result.height = depthH;
    result.depthMap =
            QVector<float>(depth, depth + static_cast<size_t>(depthW) * depthH);
    aicore_yolo_free_buffer(depth);
    result.modelVariant =
            QString::fromUtf8(aicore_yolo_context_model_name(m_pendingCtx));
    result.imageSize =
            static_cast<int>(aicore_yolo_context_image_size(m_pendingCtx));
    const char* resolvedDevice = aicore_yolo_context_device(m_pendingCtx);
    result.resolvedDevice = (resolvedDevice && resolvedDevice[0])
                                    ? QString::fromUtf8(resolvedDevice)
                                    : m_settings.device;

    // Statistics envelope (min/max/mean/p95 over valid pixels).
    if (char* statsJson = aicore_yolo_last_depth_json(m_pendingCtx)) {
        result.resultJson = QByteArray(statsJson);
        aicore_yolo_free_buffer(statsJson);
        YOLOHelpers::parseDepthStatsJson(result.resultJson, &result.stats);
    }

    // Colorized export image: turbo ramp over [min, p95] + legend. The p95
    // far bound ignores outlier sky/background pixels that would otherwise
    // crush the useful near range.
    result.annotatedImage = YOLOHelpers::depthColorImage(
            result.depthMap.constData(), result.width, result.height,
            result.stats.minDepth, result.stats.p95Depth);
    if (!result.annotatedImage.isNull()) {
        YOLOHelpers::drawDepthLegend(&result.annotatedImage,
                                     result.stats.minDepth,
                                     result.stats.p95Depth);
    }

    emit logMessage(tr("[YOLO] Depth %1x%2 in %3 ms (model=%4, range=%5-%6 m)")
                            .arg(depthW)
                            .arg(depthH)
                            .arg(ms, 0, 'f', 1)
                            .arg(result.modelVariant)
                            .arg(result.stats.minDepth, 0, 'f', 2)
                            .arg(result.stats.p95Depth, 0, 'f', 2));
    emit depthResultReady(result);
    return true;
}
#endif
