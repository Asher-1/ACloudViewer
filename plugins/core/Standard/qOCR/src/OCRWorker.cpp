// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "OCRWorker.h"

#include <QFile>
#include <QFileInfo>
#include <cstring>

#include "aicore/ocr_capi.h"
#include "ecvAICoreRuntimeHelpers.h"

OCRWorker::OCRWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    m_cancelToken = aicore_cancel_token_new();
}

OCRWorker::~OCRWorker() {
    releaseContextOnMainThread();
    if (m_cancelToken) {
        aicore_cancel_token_free(m_cancelToken);
        m_cancelToken = nullptr;
    }
}

void OCRWorker::releaseContextOnMainThread() {
    // The context is created on the worker thread; destroy it here (main
    // thread) so GPU teardown never races the render thread.
    ecvAICoreRuntime::releasePending(m_pendingCtx, &aicore_ocr_context_destroy);
}

void OCRWorker::requestTaskCancel() {
    if (m_cancelToken) aicore_cancel_token_request(m_cancelToken);
}

void OCRWorker::run() {
    // Serialize against every other AICore inference task on the same
    // device (live video loops, other plugin workers): concurrent graph
    // compute on one backend races its command queues and poisons the
    // backend for the rest of the process.
    auto taskGuard = ecvAICoreRuntime::makeDeviceTaskLock(m_settings.device);
    if (!taskGuard.isLocked()) {
        emit logMessage(
                tr("[qOCR] Failed to acquire the inference device; another "
                   "task is running."));
        emit taskFinished(false);
        return;
    }

    emit taskStage(tr("Loading models…"), -1);
    emit logMessage(
            tr("[qOCR] Loading det/rec models (device=%1, threads=%2)...")
                    .arg(m_settings.device)
                    .arg(m_settings.threads));

    aicore_ocr_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = AICORE_OCR_ABI_VERSION;
    const QByteArray detBytes = QFile::encodeName(m_settings.detPath);
    const QByteArray recBytes = QFile::encodeName(m_settings.recPath);
    const QByteArray deviceBytes = m_settings.device.toUtf8();
    options.det_model_path = detBytes.constData();
    options.rec_model_path = recBytes.constData();
    // Optional layout stage: the pack travels through the same settings
    // channel; an empty path keeps the regions array empty.
    QByteArray layoutBytes;
    if (!m_settings.layoutPath.isEmpty()) {
        layoutBytes = QFile::encodeName(m_settings.layoutPath);
        options.layout_model_path = layoutBytes.constData();
    }
    options.device = deviceBytes.constData();
    options.n_threads = m_settings.threads;
    m_pendingCtx = aicore_ocr_context_create(&options);
    if (!m_pendingCtx) {
        emit logMessage(
                tr("[qOCR] Model load failed: %1")
                        .arg(QString::fromUtf8(aicore_ocr_last_error())));
        emit taskFinished(false);
        return;
    }
    if (!aicore_ocr_context_ready(m_pendingCtx)) {
        emit logMessage(tr("[qOCR] Model context is not ready."));
        emit taskFinished(false);
        return;
    }
    const QString resolvedDevice =
            QString::fromUtf8(aicore_ocr_context_device(m_pendingCtx));
    emit logMessage(tr("[qOCR] Models loaded on device: %1")
                            .arg(resolvedDevice.isEmpty() ? m_settings.device
                                                          : resolvedDevice));

    if (aicore_cancel_token_requested(m_cancelToken)) {
        emit logMessage(tr("[qOCR] Cancelled before inference."));
        emit taskFinished(false);
        return;
    }

    // Borrowed, stride-aware image view (ecvAICoreRuntimeHelpers). The
    // converted QImage must outlive the aicore_ocr_run call below.
    QImage source(m_settings.imagePath);
    if (source.isNull()) {
        emit logMessage(tr("[qOCR] Failed to load image: %1")
                                .arg(m_settings.imagePath));
        emit taskFinished(false);
        return;
    }
    aicore_image_view view = ecvAICoreRuntime::makeImageView(source);
    if (!view.data) {
        // Unsupported source format: convert once, keep it alive for the run.
        source = source.convertToFormat(QImage::Format_RGB888);
        view = ecvAICoreRuntime::makeImageView(source);
    }
    if (!view.data) {
        emit logMessage(tr("[qOCR] Unsupported image format."));
        emit taskFinished(false);
        return;
    }

    emit taskStage(tr("Detecting and recognizing text…"), -1);
    aicore_cancel_scope_begin(m_cancelToken);
    aicore_ocr_result* result = aicore_ocr_run(m_pendingCtx, &view);
    aicore_cancel_scope_end(m_cancelToken);
    if (!result) {
        emit logMessage(
                tr("[qOCR] Inference failed: %1")
                        .arg(QString::fromUtf8(aicore_ocr_last_error())));
        emit taskFinished(false);
        return;
    }

    OCRRunResult runResult;
    runResult.sourceImage = source;
    runResult.imagePath = m_settings.imagePath;
    runResult.detModel = QFileInfo(m_settings.detPath).fileName();
    runResult.recModel = QFileInfo(m_settings.recPath).fileName();
    runResult.layoutModel =
            m_settings.layoutPath.isEmpty()
                    ? QString()
                    : QFileInfo(m_settings.layoutPath).fileName();
    runResult.resolvedDevice = resolvedDevice;
    const int count = aicore_ocr_box_count(result);
    runResult.boxes.reserve(count);
    for (int i = 0; i < count; ++i) {
        const aicore_ocr_box* box = aicore_ocr_box_at(result, i);
        if (!box) continue;
        OCRBoxResult entry;
        std::memcpy(entry.quad, box->quad, sizeof(entry.quad));
        entry.text = QString::fromUtf8(box->text);
        entry.score = box->score;
        entry.language = QString::fromLatin1(box->language);
        runResult.boxes.push_back(entry);
    }
    const int regionCount = aicore_ocr_region_count(result);
    runResult.regions.reserve(regionCount);
    for (int r = 0; r < regionCount; ++r) {
        const aicore_ocr_region* region = aicore_ocr_region_at(result, r);
        if (!region) continue;
        OCRRegionResult entry;
        entry.classId = region->class_id;
        entry.readingOrder = region->reading_order;
        std::memcpy(entry.quad, region->quad, sizeof(entry.quad));
        entry.score = region->score;
        const char* className =
                aicore_ocr_label_at(m_pendingCtx, region->class_id);
        entry.className = className ? QString::fromUtf8(className)
                                    : QString::number(region->class_id);
        runResult.regions.push_back(entry);
    }
    const aicore_pipeline_timings timings = aicore_ocr_timings(result);
    if (timings.valid_fields & AICORE_TIMING_PREPROCESS) {
        runResult.preprocessMs = timings.preprocess_ms;
    }
    if (timings.valid_fields & AICORE_TIMING_INFERENCE) {
        runResult.inferenceMs = timings.inference_ms;
    }
    if (timings.valid_fields & AICORE_TIMING_POSTPROCESS) {
        runResult.postprocessMs = timings.postprocess_ms;
    }
    if (timings.valid_fields & AICORE_TIMING_E2E) {
        runResult.e2eMs = timings.e2e_ms;
    }
    aicore_ocr_result_release(result);

    emit taskStage(tr("Done"), 100);
    emit logMessage(tr("[qOCR] %1 region(s) recognized in %2 ms [%3]")
                            .arg(count)
                            .arg(runResult.e2eMs, 0, 'f', 0)
                            .arg(runResult.resolvedDevice));
    emit resultReady(runResult);
    emit taskFinished(true);
}
