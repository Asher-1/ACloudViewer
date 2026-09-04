// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOLiveInferWorker.h"

#include <QtCompat.h>

#include <QByteArray>
#include <QElapsedTimer>
#include <QFileInfo>
#include <new>
#include <utility>

#ifdef AICore_ENABLED
#include "aicore/runtime_capi.h"
#include "aicore/yolo_capi.h"
#endif

namespace {

#ifdef AICore_ENABLED
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

aicore_image_view imageView(const QImage& image) {
    return aicore_image_view{
            reinterpret_cast<const uint8_t*>(image.constBits()), image.width(),
            image.height(), static_cast<size_t>(image.bytesPerLine()),
            AICORE_IMAGE_RGB8};
}
#endif

}  // namespace

YOLOLiveInferWorker::YOLOLiveInferWorker(QObject* parent) : QObject(parent) {
    qRegisterMetaType<YOLOLiveInferWorker::Job>("YOLOLiveInferWorker::Job");
    qRegisterMetaType<YOLOLiveInferWorker::Result>(
            "YOLOLiveInferWorker::Result");
}

YOLOLiveInferWorker::~YOLOLiveInferWorker() { releaseModel(); }

void YOLOLiveInferWorker::releaseModel() {
#ifdef AICore_ENABLED
    if (m_ctx) {
        aicore_yolo_free(m_ctx);
        m_ctx = nullptr;
    }
    m_loadedModelPath.clear();
    m_loadedDevice.clear();
    m_loadedThreads = 0;
    m_loadedTask.clear();
    m_resolvedDevice.clear();
#endif
}

#ifdef AICore_ENABLED
bool YOLOLiveInferWorker::ensureModel(const Job& job, QString* error) {
    if (job.modelPath.isEmpty() || !QFileInfo::exists(job.modelPath)) {
        if (error) *error = tr("Model file does not exist.");
        return false;
    }
    if (m_ctx && aicore_yolo_is_ready(m_ctx) &&
        m_loadedModelPath == job.modelPath && m_loadedDevice == job.device &&
        m_loadedThreads == job.threads) {
        return true;
    }

    releaseModel();
    aicore_yolo_options* opts = aicore_yolo_options_new();
    if (!opts) {
        if (error) *error = tr("Failed to allocate model options.");
        return false;
    }
    aicore_yolo_options_set_device(opts, job.device.toUtf8().constData());
    aicore_yolo_options_set_threads(opts, job.threads);
    m_ctx = aicore_yolo_load_opts(job.modelPath.toUtf8().constData(), opts);
    aicore_yolo_options_free(opts);
    if (!m_ctx || !aicore_yolo_is_ready(m_ctx)) {
        const char* message = m_ctx ? aicore_yolo_last_error(m_ctx) : nullptr;
        if (error) {
            *error = message ? QString::fromUtf8(message)
                             : tr("Failed to create model context.");
        }
        releaseModel();
        return false;
    }
    m_loadedModelPath = job.modelPath;
    m_loadedDevice = job.device;
    m_loadedThreads = job.threads;
    m_loadedTask = QString::fromUtf8(aicore_yolo_context_task(m_ctx));
    /* The backend-resolved device ("CUDA0", "cpu", ...), captured at load
     * time so every Result reports what actually ran — a requested GPU that
     * silently fell back to CPU shows up here. */
    const char* resolved = aicore_yolo_context_device(m_ctx);
    m_resolvedDevice = (resolved && resolved[0]) ? QString::fromUtf8(resolved)
                                                 : job.device;
    return true;
}
#endif

void YOLOLiveInferWorker::runJob(YOLOLiveInferWorker::Job job) {
    // The queued slot boundary: an uncaught allocation failure inside the
    // worker would unwind through the worker thread's event loop and
    // terminate the whole process (SIGABRT). Surface it as a per-frame error
    // instead so video inference keeps running.
    const quint64 generation = job.generation;
    try {
        runJobImpl(std::move(job));
    } catch (const std::bad_alloc&) {
        Result result;
        result.generation = generation;
        result.error = tr("Out of memory while processing the frame.");
        emit inferComplete(result);
    }
}

void YOLOLiveInferWorker::runJobImpl(YOLOLiveInferWorker::Job job) {
    Result result;
    result.generation = job.generation;

#ifndef AICore_ENABLED
    result.error = tr("AICore is not enabled.");
    emit inferComplete(result);
    return;
#else
    DeviceTaskGuard taskGuard(job.device);
    if (!taskGuard.isLocked()) {
        result.error = tr("Failed to acquire the inference device.");
        emit inferComplete(result);
        return;
    }
    if (!ensureModel(job, &result.error)) {
        emit inferComplete(result);
        return;
    }
    result.task = m_loadedTask;

    const aicore_image_view image = imageView(job.rgb);

    if (result.task == QStringLiteral("depth")) {
        // Metric depth: typed float map + statistics envelope. The colorized
        // image is NOT built here — the live preview only needs a downscaled
        // layer (drawn by the widget), so the full-resolution colorize pass
        // runs once at capture time.
        QElapsedTimer timer;
        timer.start();
        int32_t dw = 0, dh = 0;
        float* depth = aicore_yolo_depth_image(m_ctx, &image, &dw, &dh);
        result.depth.runtimeMs = static_cast<double>(timer.elapsed());
        if (!depth || dw <= 0 || dh <= 0) {
            if (depth) aicore_yolo_free_buffer(depth);
            const char* message = aicore_yolo_last_error(m_ctx);
            result.error = message ? QString::fromUtf8(message)
                                   : tr("YOLO depth inference failed.");
            emit inferComplete(result);
            return;
        }
        result.depth.depthMap = qtCompatQVectorFromRange<float>(
                depth, depth + static_cast<size_t>(dw) * dh);
        aicore_yolo_free_buffer(depth);
        result.depth.width = dw;
        result.depth.height = dh;
        aicore_yolo_depth_stats stats{};
        if (aicore_yolo_last_depth_stats(m_ctx, &stats) == 0) {
            result.depth.stats.width = stats.depth_width;
            result.depth.stats.height = stats.depth_height;
            result.depth.stats.minDepth = stats.min_depth;
            result.depth.stats.maxDepth = stats.max_depth;
            result.depth.stats.meanDepth = stats.mean_depth;
            result.depth.stats.p95Depth = stats.p95_depth;
            result.depth.stats.validPixels =
                    static_cast<long long>(stats.valid_pixels);
        }
        result.depth.modelPath = job.modelPath;
        result.depth.resolvedDevice = m_resolvedDevice;
        result.depth.imageName = QStringLiteral("live");
        result.ok = true;
        emit inferComplete(result);
        return;
    }

    if (result.task == QStringLiteral("segment")) {
        // Instance segmentation: typed detections + per-instance masks.
        QElapsedTimer timer;
        timer.start();
        aicore_yolo_set_detect_thresholds(m_ctx, job.confThres, job.iouThres,
                                          job.topK);
        aicore_yolo_segment_result* seg = aicore_yolo_seg_image(m_ctx, &image);
        result.detect.runtimeMs = static_cast<double>(timer.elapsed());
        if (!seg) {
            const char* message = aicore_yolo_last_error(m_ctx);
            result.error = message ? QString::fromUtf8(message)
                                   : tr("YOLO segmentation failed.");
            emit inferComplete(result);
            return;
        }

        const int n = aicore_yolo_seg_det_count(seg);
        result.detect.detections.reserve(n > 0 ? n : 0);
        result.detect.masks.reserve(n > 0 ? n : 0);
        for (int i = 0; i < n; ++i) {
            const aicore_yolo_detection det = aicore_yolo_seg_det_at(seg, i);
            YOLODetection d;
            d.classId = det.class_id;
            d.x1 = det.x1;
            d.y1 = det.y1;
            d.x2 = det.x2;
            d.y2 = det.y2;
            d.score = det.score;
            result.detect.detections.append(d);

            const aicore_yolo_plane_view view = aicore_yolo_seg_mask_at(seg, i);
            if (view.data != nullptr && view.width > 0 && view.height > 0) {
                YOLOSegMask mask;
                mask.w = view.width;
                mask.h = view.height;
                mask.bits = QByteArray(
                        static_cast<const char*>(view.data),
                        static_cast<int>(view.row_stride_bytes) * view.height);
                result.detect.masks.append(mask);
            }
        }
        for (int i = 0; i < result.detect.detections.size(); ++i) {
            const char* name = aicore_yolo_seg_det_class_name(seg, i);
            result.detect.detections[i].className =
                    (name != nullptr && name[0] != '\0')
                            ? QString::fromUtf8(name)
                            : QStringLiteral("class %1")
                                      .arg(result.detect.detections[i].classId);
        }
        result.detect.totalDetected = n;
        result.detect.task = QStringLiteral("segment");
        aicore_yolo_seg_result_free(seg);

        result.detect.modelPath = job.modelPath;
        result.detect.resolvedDevice = m_resolvedDevice;
        result.detect.imageName = QStringLiteral("live");
        result.ok = true;
        emit inferComplete(result);
        return;
    }

    // Detect: typed context-owned records; JSON is reserved for explicit API
    // compatibility callers and never crosses the live-video hot path.
    QElapsedTimer timer;
    timer.start();
    aicore_yolo_set_detect_thresholds(m_ctx, job.confThres, job.iouThres,
                                      job.topK);
    const int detectRc = aicore_yolo_detect_image(m_ctx, &image);
    result.detect.runtimeMs = static_cast<double>(timer.elapsed());
    if (detectRc != 0) {
        const char* message = aicore_yolo_last_error(m_ctx);
        result.error = message ? QString::fromUtf8(message)
                               : tr("YOLO inference failed.");
        emit inferComplete(result);
        return;
    }

    const int count = aicore_yolo_detection_count(m_ctx);
    result.detect.detections.reserve(std::max(0, count));
    for (int i = 0; i < count; ++i) {
        const aicore_yolo_detection det = aicore_yolo_detection_at(m_ctx, i);
        YOLODetection out;
        out.classId = static_cast<uint32_t>(det.class_id);
        // Backend class table (open-vocabulary class-list override or the
        // GGUF metadata); fall back to the deterministic label only when the
        // model declares no name for this class.
        const char* name = aicore_yolo_detection_class_name(m_ctx, i);
        out.className = (name != nullptr && name[0] != '\0')
                                ? QString::fromUtf8(name)
                                : QStringLiteral("class %1").arg(det.class_id);
        out.x1 = det.x1;
        out.y1 = det.y1;
        out.x2 = det.x2;
        out.y2 = det.y2;
        out.score = det.score;
        result.detect.detections.append(out);
    }
    result.detect.totalDetected = count;

    // No annotated rendering here: the live preview only needs a downscaled
    // overlay (drawn by the widget on the display image), so a per-frame
    // full-resolution drawDetections pass would be pure waste. The widget
    // caches the submitted frame and renders annotatedImage once at capture
    // time from the cached detections / depth map.

    result.detect.modelPath = job.modelPath;
    /* The device the backend actually resolved to (may differ from
     * job.device when the GPU lease failed and yolo fell back to CPU). */
    result.detect.resolvedDevice = m_resolvedDevice;
    result.detect.imageName = QStringLiteral("live");
    result.ok = true;
    emit inferComplete(result);
#endif
}
