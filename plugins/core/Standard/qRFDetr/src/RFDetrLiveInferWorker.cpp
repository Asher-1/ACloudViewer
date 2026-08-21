// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RFDetrLiveInferWorker.h"

#include <QByteArray>
#include <QElapsedTimer>
#include <QFileInfo>
#include <algorithm>
#include <new>
#include <utility>

#ifdef AICore_ENABLED
#include "aicore/rfdetr_capi.h"
#include "aicore/runtime_capi.h"
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
#endif

}  // namespace

RFDetrLiveInferWorker::RFDetrLiveInferWorker(QObject* parent)
    : QObject(parent) {
    qRegisterMetaType<RFDetrLiveInferWorker::Job>("RFDetrLiveInferWorker::Job");
    qRegisterMetaType<RFDetrLiveInferWorker::Result>(
            "RFDetrLiveInferWorker::Result");
}

RFDetrLiveInferWorker::~RFDetrLiveInferWorker() { releaseModel(); }

void RFDetrLiveInferWorker::releaseModel() {
#ifdef AICore_ENABLED
    if (m_ctx) {
        aicore_rfdetr_free(m_ctx);
        m_ctx = nullptr;
    }
    m_loadedModelPath.clear();
    m_loadedDevice.clear();
    m_loadedThreads = 0;
    m_resolvedDevice.clear();
#endif
}

#ifdef AICore_ENABLED
bool RFDetrLiveInferWorker::ensureModel(const Job& job, QString* error) {
    if (job.modelPath.isEmpty() || !QFileInfo::exists(job.modelPath)) {
        if (error) *error = tr("Model file does not exist.");
        return false;
    }
    if (m_ctx && aicore_rfdetr_is_ready(m_ctx) &&
        m_loadedModelPath == job.modelPath && m_loadedDevice == job.device &&
        m_loadedThreads == job.threads) {
        return true;
    }

    releaseModel();
    aicore_rfdetr_options* opts = aicore_rfdetr_options_new();
    if (!opts) {
        if (error) *error = tr("Failed to allocate model options.");
        return false;
    }
    aicore_rfdetr_options_set_device(opts, job.device.toUtf8().constData());
    aicore_rfdetr_options_set_threads(opts, job.threads);
    m_ctx = aicore_rfdetr_load_opts(job.modelPath.toUtf8().constData(), opts);
    aicore_rfdetr_options_free(opts);
    if (!m_ctx || !aicore_rfdetr_is_ready(m_ctx)) {
        const char* message = m_ctx ? aicore_rfdetr_last_error(m_ctx) : nullptr;
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
    /* The backend-resolved device ("CUDA0", "cpu", ...), captured at load
     * time so every Result reports what actually ran — a requested GPU that
     * silently fell back to CPU shows up here. */
    const char* resolved = aicore_rfdetr_context_device(m_ctx);
    m_resolvedDevice = (resolved && resolved[0]) ? QString::fromUtf8(resolved)
                                                 : job.device;
    return true;
}
#endif

void RFDetrLiveInferWorker::runJob(RFDetrLiveInferWorker::Job job) {
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

void RFDetrLiveInferWorker::runJobImpl(RFDetrLiveInferWorker::Job job) {
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

    QByteArray packedRgb;
    const uchar* rgb = RFDetrHelpers::packedRgb888Data(job.rgb, &packedRgb);
    if (!rgb) {
        result.error = tr("Live frame is not RGB888.");
        emit inferComplete(result);
        return;
    }

    QElapsedTimer timer;
    timer.start();
    char* json = aicore_rfdetr_detect_rgb_json(m_ctx, rgb, job.rgb.width(),
                                               job.rgb.height(), job.threshold,
                                               job.topK);
    result.snapshot.runtimeMs = static_cast<double>(timer.elapsed());
    if (!json) {
        const char* message = aicore_rfdetr_last_error(m_ctx);
        result.error = message ? QString::fromUtf8(message)
                               : tr("RF-DETR inference failed.");
        emit inferComplete(result);
        return;
    }

    const QByteArray payload(json);
    aicore_rfdetr_free_buffer(json);
    if (!RFDetrHelpers::parseDetectionsJson(payload, &result.snapshot)) {
        result.error = tr("Failed to parse detection output.");
        emit inferComplete(result);
        return;
    }

    if (aicore_rfdetr_context_has_segmentation(m_ctx)) {
        const int count =
                std::min(aicore_rfdetr_detection_count(m_ctx),
                         static_cast<int>(result.snapshot.detections.size()));
        for (int i = 0; i < count; ++i) {
            // Raw mask bytes (no PNG encode/decode round-trip per frame);
            // sizing call also returns the mask dimensions.
            int32_t mw = 0, mh = 0;
            const int length = aicore_rfdetr_detection_mask(m_ctx, i, nullptr,
                                                            0, &mw, &mh);
            if (length <= 0) continue;
            QByteArray raw;
            raw.resize(length);
            if (aicore_rfdetr_detection_mask(
                        m_ctx, i, reinterpret_cast<unsigned char*>(raw.data()),
                        length, &mw, &mh) == length) {
                result.snapshot.detections[i].maskRaw = raw;
                result.snapshot.detections[i].maskWidth = mw;
                result.snapshot.detections[i].maskHeight = mh;
            }
        }
    }

    // No annotated rendering here: the live preview only needs a downscaled
    // overlay (drawn by the widget on the display image), so a per-frame
    // full-resolution drawDetections pass would be pure waste. The widget
    // caches the submitted frame and renders annotatedImage once at capture
    // time from snapshot.detections.

    result.snapshot.modelPath = job.modelPath;
    /* The device the backend actually resolved to (may differ from
     * job.device when the GPU lease failed and rfdetr fell back to CPU). */
    result.snapshot.resolvedDevice = m_resolvedDevice;
    result.snapshot.imageName = QStringLiteral("live");
    result.ok = true;
    emit inferComplete(result);
#endif
}
