// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOWorker.h"

#include <QtCompat.h>

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QImage>
#include <QStringList>
#include <algorithm>
#include <vector>

#ifdef AICore_ENABLED
#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"
#endif

namespace {

#ifdef AICore_ENABLED
/* Serializes this worker against every other AICore inference task on the
 * same device (live video loops, other plugin workers). ggml-metal's backend
 * state machine is not safe under concurrent graph compute from two threads;
 * the shared device queue lock is the process-wide mutex that keeps command
 * buffers from racing (a failed command buffer poisons the backend for the
 * rest of the process). */
class DeviceTaskGuard {
public:
    explicit DeviceTaskGuard(const QString& device)
        : m_locked(aicore_device_task_lock(device.toUtf8().constData()) == 0) {}
    ~DeviceTaskGuard() {
        if (m_locked) aicore_device_task_unlock();
    }
    bool isLocked() const { return m_locked; }

private:
    bool m_locked = false;
};

aicore_image_format imageFormat(const QImage& image) {
    switch (image.format()) {
        case QImage::Format_RGB888:
            return AICORE_IMAGE_RGB8;
        case QImage::Format_RGBA8888:
            return AICORE_IMAGE_RGBA8;
        case QImage::Format_Grayscale8:
            return AICORE_IMAGE_GRAY8;
#if QT_VERSION >= QT_VERSION_CHECK(5, 14, 0)
        case QImage::Format_BGR888:
            return AICORE_IMAGE_BGR8;
#endif
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
        case QImage::Format_RGB32:
        case QImage::Format_ARGB32:
            return AICORE_IMAGE_BGRA8;
#endif
        default:
            return static_cast<aicore_image_format>(0);
    }
}

QImage inferenceImage(const QImage& image) {
    return imageFormat(image) != 0
                   ? image
                   : image.convertToFormat(QImage::Format_RGB888);
}

aicore_image_view imageView(const QImage& image) {
    return aicore_image_view{
            reinterpret_cast<const uint8_t*>(image.constBits()), image.width(),
            image.height(), static_cast<size_t>(image.bytesPerLine()),
            imageFormat(image)};
}
#endif

}  // namespace

namespace {

/* True when the class list carries a multi-word descriptive phrase — the
 * open-vocabulary heads score those far below short category nouns, so an
 * empty result with such a prompt is almost always a threshold/wording
 * issue, not a broken pipeline. */
bool hasPhrasePrompt(const QStringList& classes) {
    for (const QString& c : classes) {
        if (c.contains(QLatin1Char(' '))) return true;
    }
    return false;
}

}  // namespace

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
    DeviceTaskGuard taskGuard(m_settings.device);
    if (!taskGuard.isLocked()) {
        emit logMessage(
                tr("[YOLO] Failed to acquire the inference device; another "
                   "task is running."));
        return false;
    }
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
    // Open-vocabulary setup (world/yoloe): the class list rides into load;
    // the text-model GGUF encodes it once per context. YOLOE visual prompts
    // take precedence: the SAVPE encoder derives the class embeddings from
    // the drawn boxes, so no class list and no text model apply.
    if (!m_settings.visualPrompts.isEmpty()) {
        std::vector<float> boxes;
        boxes.reserve(static_cast<size_t>(m_settings.visualPrompts.size()) * 4);
        for (const QRectF& b : m_settings.visualPrompts) {
            boxes.push_back(static_cast<float>(b.left()));
            boxes.push_back(static_cast<float>(b.top()));
            boxes.push_back(static_cast<float>(b.right()));
            boxes.push_back(static_cast<float>(b.bottom()));
        }
        aicore_yolo_options_set_visual_prompts(
                opts, boxes.data(),
                static_cast<int32_t>(m_settings.visualPrompts.size()));
        const int named =
                std::count_if(m_settings.visualPromptNames.cbegin(),
                              m_settings.visualPromptNames.cend(),
                              [](const QString& n) { return !n.isEmpty(); });
        emit logMessage(
                named > 0 ? tr("[YOLO] Visual prompts: %1 example "
                               "box(es); %2 named (SAVPE derives the "
                               "categories from the boxes).")
                                    .arg(m_settings.visualPrompts.size())
                                    .arg(named)
                          : tr("[YOLO] Visual prompts: %1 example "
                               "box(es); the SAVPE encoder derives the "
                               "categories (object0..object%2). Double-"
                               "click a box on the canvas to name it.")
                                    .arg(m_settings.visualPrompts.size())
                                    .arg(m_settings.visualPrompts.size() - 1));
    } else if (!m_settings.classes.isEmpty()) {
        std::vector<const char*> classPtrs;
        classPtrs.reserve(static_cast<size_t>(m_settings.classes.size()));
        QList<QByteArray> utf8;
        utf8.reserve(m_settings.classes.size());
        for (const QString& c : m_settings.classes) {
            utf8.append(c.toUtf8());
        }
        for (const QByteArray& c : utf8) {
            classPtrs.push_back(c.constData());
        }
        aicore_yolo_options_set_classes(opts, classPtrs.data(),
                                        static_cast<int32_t>(classPtrs.size()));
        if (!m_settings.textModelPath.isEmpty()) {
            aicore_yolo_options_set_text_model(
                    opts, m_settings.textModelPath.toUtf8().constData());
        }
    }

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
    const QImage rgb = inferenceImage(input);
    emit progressUpdate(0, 1);

    // The loaded model decides the path: a detect GGUF yields boxes, a
    // segment GGUF yields boxes + instance masks, a depth GGUF yields a
    // metric depth map, pose/obb/semantic/classify GGUFs yield their typed
    // results — there is no user-side task switch that could disagree with
    // the model. World/YOLOE models are text-conditioned detect/segment
    // variants and flow through the same paths (the class vocabulary comes
    // from the tab's class list + text model).
    const bool ok = (task == QStringLiteral("depth"))      ? runDepth(rgb)
                    : (task == QStringLiteral("segment"))  ? runSegment(rgb)
                    : (task == QStringLiteral("pose"))     ? runPose(rgb)
                    : (task == QStringLiteral("obb"))      ? runObb(rgb)
                    : (task == QStringLiteral("semantic")) ? runSemantic(rgb)
                    : (task == QStringLiteral("classify")) ? runClassify(rgb)
                                                           : runDetect(rgb);
    emit progressUpdate(1, 1);
    return ok;
}

QString YOLOWorker::visualPromptName(uint32_t classId) const {
    if (classId < static_cast<uint32_t>(m_settings.visualPromptNames.size())) {
        return m_settings.visualPromptNames.at(static_cast<int>(classId));
    }
    return QString();
}

bool YOLOWorker::runDetect(const QImage& rgb) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    const aicore_image_view image = imageView(rgb);
    const int detectRc = aicore_yolo_detect_image(m_pendingCtx, &image);
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (detectRc != 0) {
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
    result.modelVariant =
            QString::fromUtf8(aicore_yolo_context_model_name(m_pendingCtx));
    result.imageSize =
            static_cast<int>(aicore_yolo_context_image_size(m_pendingCtx));
    result.numClasses =
            static_cast<int>(aicore_yolo_context_num_classes(m_pendingCtx));
    result.end2end = aicore_yolo_context_end2end(m_pendingCtx) != 0;
    const int detectionCount = aicore_yolo_detection_count(m_pendingCtx);
    result.detections.reserve(detectionCount > 0 ? detectionCount : 0);
    for (int i = 0; i < detectionCount; ++i) {
        const aicore_yolo_detection det =
                aicore_yolo_detection_at(m_pendingCtx, i);
        YOLODetection d;
        d.classId = static_cast<uint32_t>(det.class_id);
        d.className = QStringLiteral("class %1").arg(det.class_id);
        // Visual-prompt naming: a user-assigned prompt name replaces the
        // positional class-N/objectN label (detections are labeled by the
        // prompt index in visual-prompt mode).
        const QString promptName = visualPromptName(d.classId);
        if (!promptName.isEmpty()) d.className = promptName;
        d.x1 = det.x1;
        d.y1 = det.y1;
        d.x2 = det.x2;
        d.y2 = det.y2;
        d.score = det.score;
        result.detections.append(d);
    }
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
    if (result.detections.isEmpty() && hasPhrasePrompt(m_settings.classes)) {
        emit logMessage(
                tr("[YOLO] Hint: descriptive phrase prompts (e.g. \"female "
                   "in yellow hat\") score far below short categories on "
                   "open-vocabulary heads — split into short nouns (person, "
                   "hat) or lower Confidence to ~0.02."));
    }
    emit resultReady(result);
    return true;
}

bool YOLOWorker::runSegment(const QImage& rgb) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    const aicore_image_view image = imageView(rgb);
    aicore_yolo_segment_result* seg =
            aicore_yolo_seg_image(m_pendingCtx, &image);
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
        // Visual-prompt naming: a user-assigned prompt name replaces the
        // backend's positional objectN label (class_id = prompt index in
        // visual-prompt mode).
        const QString promptName =
                visualPromptName(result.detections[i].classId);
        if (!promptName.isEmpty()) {
            result.detections[i].className = promptName;
        }
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
    if (n == 0 && hasPhrasePrompt(m_settings.classes)) {
        emit logMessage(
                tr("[YOLO] Hint: descriptive phrase prompts (e.g. \"female "
                   "in yellow hat\") score far below short categories on "
                   "open-vocabulary heads — split into short nouns (person, "
                   "hat) or lower Confidence to ~0.02."));
    }
    emit resultReady(result);
    return true;
}

bool YOLOWorker::runDepth(const QImage& rgb) {
    QElapsedTimer timer;
    timer.start();
    int32_t depthW = 0, depthH = 0;
    aicore_cancel_scope_begin(m_cancelToken);
    const aicore_image_view image = imageView(rgb);
    float* depth =
            aicore_yolo_depth_image(m_pendingCtx, &image, &depthW, &depthH);
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
    result.depthMap = qtCompatQVectorFromRange<float>(
            depth, depth + static_cast<size_t>(depthW) * depthH);
    aicore_yolo_free_buffer(depth);
    result.modelVariant =
            QString::fromUtf8(aicore_yolo_context_model_name(m_pendingCtx));
    result.imageSize =
            static_cast<int>(aicore_yolo_context_image_size(m_pendingCtx));
    const char* resolvedDevice = aicore_yolo_context_device(m_pendingCtx);
    result.resolvedDevice = (resolvedDevice && resolvedDevice[0])
                                    ? QString::fromUtf8(resolvedDevice)
                                    : m_settings.device;

    aicore_yolo_depth_stats stats{};
    if (aicore_yolo_last_depth_stats(m_pendingCtx, &stats) == 0) {
        result.stats.width = stats.depth_width;
        result.stats.height = stats.depth_height;
        result.stats.minDepth = stats.min_depth;
        result.stats.maxDepth = stats.max_depth;
        result.stats.meanDepth = stats.mean_depth;
        result.stats.p95Depth = stats.p95_depth;
        result.stats.validPixels = static_cast<long long>(stats.valid_pixels);
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

bool YOLOWorker::runPose(const QImage& rgb) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    const aicore_image_view image = imageView(rgb);
    aicore_yolo_pose_result* pose =
            aicore_yolo_pose_image(m_pendingCtx, &image);
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (!pose) {
        const char* err = aicore_yolo_last_error(m_pendingCtx);
        emit logMessage(tr("[YOLO] Pose inference failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    YOLORunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.modelPath = m_settings.modelPath;
    result.task = QStringLiteral("pose");
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

    const int n = aicore_yolo_pose_det_count(pose);
    result.kptCount = aicore_yolo_pose_kpt_count(pose);
    result.keypointSets.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_detection det = aicore_yolo_pose_det_at(pose, i);
        YOLOKeypointSet set;
        set.det.classId = det.class_id;
        set.det.x1 = det.x1;
        set.det.y1 = det.y1;
        set.det.x2 = det.x2;
        set.det.y2 = det.y2;
        set.det.score = det.score;
        for (int k = 0; k < result.kptCount; ++k) {
            const aicore_yolo_keypoint kp = aicore_yolo_pose_kpt_at(pose, i, k);
            set.kpts.append({kp.x, kp.y, kp.visibility});
        }
        const char* name = aicore_yolo_pose_det_class_name(pose, i);
        set.det.className =
                (name != nullptr && name[0] != '\0')
                        ? QString::fromUtf8(name)
                        : QStringLiteral("class %1").arg(det.class_id);
        result.keypointSets.append(set);
    }
    result.totalDetected = n;
    aicore_yolo_pose_result_free(pose);

    QImage annotated = rgb;
    YOLOHelpers::drawPose(&annotated, result.keypointSets);
    result.annotatedImage = annotated;

    emit logMessage(tr("[YOLO] %1 pose(s) in %2 ms (model=%3, kpts=%4)")
                            .arg(n)
                            .arg(ms, 0, 'f', 1)
                            .arg(result.modelVariant)
                            .arg(result.kptCount));
    emit resultReady(result);
    return true;
}

bool YOLOWorker::runObb(const QImage& rgb) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    const aicore_image_view image = imageView(rgb);
    aicore_yolo_obb_result* obb = aicore_yolo_obb_image(m_pendingCtx, &image);
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (!obb) {
        const char* err = aicore_yolo_last_error(m_pendingCtx);
        emit logMessage(tr("[YOLO] OBB inference failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    YOLORunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.modelPath = m_settings.modelPath;
    result.task = QStringLiteral("obb");
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

    const int n = aicore_yolo_obb_count(obb);
    result.obbBoxes.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_obb_box b = aicore_yolo_obb_at(obb, i);
        YOLOObbBox box;
        box.cx = b.cx;
        box.cy = b.cy;
        box.w = b.w;
        box.h = b.h;
        box.angle = b.angle;
        box.score = b.score;
        box.classId = static_cast<uint32_t>(b.class_id);
        const char* name = aicore_yolo_obb_class_name(obb, i);
        box.className = (name != nullptr && name[0] != '\0')
                                ? QString::fromUtf8(name)
                                : QStringLiteral("class %1").arg(b.class_id);
        result.obbBoxes.append(box);
    }
    result.totalDetected = n;
    aicore_yolo_obb_result_free(obb);

    QImage annotated = rgb;
    YOLOHelpers::drawObb(&annotated, result.obbBoxes);
    result.annotatedImage = annotated;

    emit logMessage(tr("[YOLO] %1 oriented box(es) in %2 ms (model=%3)")
                            .arg(n)
                            .arg(ms, 0, 'f', 1)
                            .arg(result.modelVariant));
    if (n == 0) {
        emit logMessage(
                tr("[YOLO] OBB models are trained on DOTA aerial imagery — "
                   "natural photos may legitimately yield 0 detections. Try "
                   "an aerial/satellite image for this model family."));
    }
    emit resultReady(result);
    return true;
}

bool YOLOWorker::runSemantic(const QImage& rgb) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    const aicore_image_view image = imageView(rgb);
    aicore_yolo_semantic_result* sem =
            aicore_yolo_semantic_image(m_pendingCtx, &image);
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (!sem) {
        const char* err = aicore_yolo_last_error(m_pendingCtx);
        emit logMessage(tr("[YOLO] Semantic inference failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    YOLORunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.modelPath = m_settings.modelPath;
    result.task = QStringLiteral("semantic");
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

    const aicore_yolo_plane_view view = aicore_yolo_semantic_class_map(sem);
    if (view.data != nullptr && view.width > 0 && view.height > 0) {
        result.semanticWidth = view.width;
        result.semanticHeight = view.height;
        result.semanticNumClasses = aicore_yolo_semantic_num_classes(sem);
        // Deep copy of the class map (the result is freed below).
        result.semanticClassMap = QByteArray(
                static_cast<const char*>(view.data),
                static_cast<int>(view.row_stride_bytes) * view.height);
    }
    aicore_yolo_semantic_result_free(sem);

    QImage annotated = rgb;
    YOLOHelpers::drawSemantic(&annotated, result.semanticClassMap,
                              result.semanticWidth, result.semanticHeight,
                              result.semanticNumClasses);
    result.annotatedImage = annotated;

    emit logMessage(
            tr("[YOLO] Semantic map %1x%2 (%3 classes) in %4 ms (model=%5)")
                    .arg(result.semanticWidth)
                    .arg(result.semanticHeight)
                    .arg(result.semanticNumClasses)
                    .arg(ms, 0, 'f', 1)
                    .arg(result.modelVariant));
    emit resultReady(result);
    return true;
}

bool YOLOWorker::runClassify(const QImage& rgb) {
    QElapsedTimer timer;
    timer.start();
    aicore_cancel_scope_begin(m_cancelToken);
    const aicore_image_view image = imageView(rgb);
    aicore_yolo_classify_result* cls =
            aicore_yolo_classify_image(m_pendingCtx, &image);
    aicore_cancel_scope_end(m_cancelToken);
    const double ms = static_cast<double>(timer.elapsed());

    if (!cls) {
        const char* err = aicore_yolo_last_error(m_pendingCtx);
        emit logMessage(tr("[YOLO] Classification failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    YOLORunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).fileName();
    result.modelPath = m_settings.modelPath;
    result.task = QStringLiteral("classify");
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

    const int n = aicore_yolo_classify_count(cls);
    result.classifications.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        YOLOClassProb cp;
        cp.classId = static_cast<uint32_t>(i);
        cp.prob = aicore_yolo_classify_prob_at(cls, i);
        const char* name = aicore_yolo_classify_class_name(cls, i);
        cp.className = (name != nullptr && name[0] != '\0')
                               ? QString::fromUtf8(name)
                               : QStringLiteral("class %1").arg(i);
        result.classifications.append(cp);
    }
    result.totalDetected = n;
    aicore_yolo_classify_result_free(cls);

    QImage annotated = rgb;
    YOLOHelpers::drawClassifications(&annotated, result.classifications);
    result.annotatedImage = annotated;

    emit logMessage(tr("[YOLO] %1 classes in %2 ms (model=%3)")
                            .arg(n)
                            .arg(ms, 0, 'f', 1)
                            .arg(result.modelVariant));
    emit resultReady(result);
    return true;
}
#endif
