// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RMBGLiveInferWorker.h"

#include <QByteArray>
#include <QFileInfo>
#include <new>
#include <utility>

#ifdef AICore_ENABLED
#include "aicore/rmbg_capi.h"
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

RMBGLiveInferWorker::RMBGLiveInferWorker(QObject* parent) : QObject(parent) {
    qRegisterMetaType<RMBGLiveInferWorker::Job>("RMBGLiveInferWorker::Job");
    qRegisterMetaType<RMBGLiveInferWorker::Result>(
            "RMBGLiveInferWorker::Result");
}

RMBGLiveInferWorker::~RMBGLiveInferWorker() { releaseModel(); }

void RMBGLiveInferWorker::releaseModel() {
#ifdef AICore_ENABLED
    if (m_ctx) {
        aicore_rmbg_free(m_ctx);
        m_ctx = nullptr;
    }
    m_loadedModelPath.clear();
    m_loadedDevice.clear();
    m_loadedThreads = 0;
#endif
}

#ifdef AICore_ENABLED
bool RMBGLiveInferWorker::ensureModel(const Job& job, QString* error) {
    if (job.modelPath.isEmpty() || !QFileInfo::exists(job.modelPath)) {
        if (error) *error = tr("Model file does not exist.");
        return false;
    }
    if (m_ctx && aicore_rmbg_is_ready(m_ctx) &&
        m_loadedModelPath == job.modelPath && m_loadedDevice == job.device &&
        m_loadedThreads == job.threads) {
        return true;
    }

    releaseModel();
    aicore_rmbg_options* opts = aicore_rmbg_options_new();
    if (!opts) {
        if (error) *error = tr("Failed to allocate model options.");
        return false;
    }
    aicore_rmbg_options_set_device(opts, job.device.toUtf8().constData());
    aicore_rmbg_options_set_threads(opts, job.threads);
    m_ctx = aicore_rmbg_load_opts(job.modelPath.toUtf8().constData(), opts);
    aicore_rmbg_options_free(opts);
    if (!m_ctx || !aicore_rmbg_is_ready(m_ctx)) {
        const char* message = m_ctx ? aicore_rmbg_last_error(m_ctx) : nullptr;
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

    // Model metadata is static for a loaded context — parse it once here
    // instead of on every frame (it used to cost a JSON serialize + parse
    // per inference).
    m_info = RMBGRunResult();
    char* info = aicore_rmbg_info_json(m_ctx);
    if (info) {
        RMBGHelpers::parseInfoJson(QByteArray(info), &m_info);
        aicore_rmbg_free_string(info);
    }
    return true;
}
#endif

void RMBGLiveInferWorker::runJob(RMBGLiveInferWorker::Job job) {
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

void RMBGLiveInferWorker::runJobImpl(RMBGLiveInferWorker::Job job) {
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

    // Static model metadata (parsed once at load) — a shallow field copy,
    // no per-frame JSON round-trip.
    result.snapshot.modelVariant = m_info.modelVariant;
    result.snapshot.inputSize = m_info.inputSize;
    result.snapshot.backend = m_info.backend;
    result.snapshot.mathProfile = m_info.mathProfile;

    QByteArray packedRgb;
    const uchar* rgb = RMBGHelpers::packedRgb888Data(job.rgb, &packedRgb);
    if (!rgb) {
        result.error = tr("Live frame is not RGB888.");
        emit inferComplete(result);
        return;
    }

    uint8_t* rgba = nullptr;
    int32_t outW = 0, outH = 0;
    int rgbaLength = 0;
    const int rc = aicore_rmbg_remove_background_rgba(
            m_ctx, rgb, job.rgb.width(), job.rgb.height(), &rgba, &outW, &outH,
            &rgbaLength);
    if (rc != 0 || !rgba || rgbaLength <= 0) {
        const char* message = aicore_rmbg_last_error(m_ctx);
        result.error = message ? QString::fromUtf8(message)
                               : tr("RMBG inference failed.");
        if (rgba) aicore_rmbg_free_buffer(rgba);
        emit inferComplete(result);
        return;
    }
    aicore_rmbg_timings timings{};
    if (aicore_rmbg_last_timings(m_ctx, &timings) == 0) {
        result.snapshot.preprocessMs = timings.preprocess_ms;
        result.snapshot.runtimeMs = timings.inference_ms;
        result.snapshot.postprocessMs = timings.postprocess_ms;
        result.snapshot.totalRuntimeMs = timings.total_ms;
    }

    // Raw RGBA composite at original resolution — no PNG round-trip (a PNG
    // encode+decode round-trip was ~50 ms per frame). QImage takes over the
    // malloc'd API buffer via a free() cleanup function, so no second
    // full-frame copy; converting once to ARGB32 up front lets alpha
    // thresholding, stats and the checkerboard preview below all reuse the
    // same buffer without further per-frame conversions.
    QImage wrapped(
            rgba, outW, outH, outW * 4, QImage::Format_RGBA8888,
            [](void* info) { std::free(info); }, rgba);
    QImage foreground = wrapped.convertToFormat(QImage::Format_ARGB32);
    if (foreground.isNull()) {
        result.error = tr("Failed to wrap the RMBG result.");
        emit inferComplete(result);
        return;
    }
    if (job.alphaThreshold > 0.0f) {
        foreground = RMBGHelpers::applyAlphaThreshold(foreground,
                                                      job.alphaThreshold);
    }
    RMBGHelpers::computeAlphaStats(foreground, &result.snapshot.alphaMean,
                                   &result.snapshot.foregroundRatio);
    result.snapshot.resultImage = foreground;
    result.snapshot.modelPath = job.modelPath;
    result.snapshot.resolvedDevice = job.device;
    result.snapshot.imageName = QStringLiteral("live");
    // The display side composites at preview resolution from the alpha of
    // resultImage (see RMBGLiveWidget::applyLiveComposite) — no full-res
    // checkerboard render per inference frame.
    result.ok = true;
    emit inferComplete(result);
#endif
}
