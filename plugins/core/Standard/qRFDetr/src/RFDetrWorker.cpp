// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RFDetrWorker.h"

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QImage>

#ifdef AICore_ENABLED
#include "aicore/rfdetr_capi.h"
#include "aicore/runtime_capi.h"
#endif

RFDetrWorker::RFDetrWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
}

RFDetrWorker::~RFDetrWorker() {
    releaseContextOnMainThread();
#ifdef AICore_ENABLED
    if (m_cancelToken) {
        aicore_cancel_token_free(m_cancelToken);
        m_cancelToken = nullptr;
    }
#endif
}

void RFDetrWorker::releaseContextOnMainThread() {
    // The context is created on the worker thread; destroy it here (main
    // thread) so GPU teardown never races the render thread.
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_rfdetr_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

void RFDetrWorker::requestTaskCancel() {
#ifdef AICore_ENABLED
    if (m_cancelToken) aicore_cancel_token_request(m_cancelToken);
#endif
}

void RFDetrWorker::run() {
#ifdef AICore_ENABLED
    const bool ok = runInference();
    emit taskFinished(ok);
#else
    emit logMessage(tr("[RF-DETR] AICore is not enabled in this build."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED
bool RFDetrWorker::runInference() {
    // Warm up the backend on the UI thread is the caller's job; here we just
    // create the model context and run.
    aicore_rfdetr_options* opts = aicore_rfdetr_options_new();
    if (!opts) {
        emit logMessage(tr("[RF-DETR] Failed to allocate options."));
        return false;
    }
    aicore_rfdetr_options_set_device(opts, m_settings.device.toUtf8().constData());
    aicore_rfdetr_options_set_threads(opts, m_settings.threads);

    emit logMessage(tr("[RF-DETR] Loading model: %1 (device=%2, threads=%3)")
                            .arg(QFileInfo(m_settings.modelPath).fileName(),
                                 m_settings.device)
                            .arg(m_settings.threads));
    emit progressUpdate(0, 1);

    m_pendingCtx = aicore_rfdetr_load_opts(
            m_settings.modelPath.toUtf8().constData(), opts);
    aicore_rfdetr_options_free(opts);
    if (!m_pendingCtx || !aicore_rfdetr_is_ready(m_pendingCtx)) {
        const char* err = m_pendingCtx ? aicore_rfdetr_last_error(m_pendingCtx)
                                       : "context allocation failed";
        emit logMessage(tr("[RF-DETR] Model load failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    emit logMessage(tr("[RF-DETR] Model loaded: variant=%1, classes=%2")
                            .arg(QString::fromUtf8(
                                    aicore_rfdetr_context_variant(m_pendingCtx)))
                            .arg(aicore_rfdetr_context_num_classes(
                                    m_pendingCtx)));
    emit progressUpdate(1, 1);

    if (aicore_cancel_token_requested(m_cancelToken)) {
        emit logMessage(tr("[RF-DETR] Cancelled before inference."));
        return false;
    }

    // Single-image inference.
    {
        const QImage input(m_settings.inputPath);
        if (input.isNull()) {
            emit logMessage(tr("[RF-DETR] Failed to load image: %1")
                                    .arg(m_settings.inputPath));
            return false;
        }
        const QImage rgb = input.convertToFormat(QImage::Format_RGB888);
        emit progressUpdate(0, 1);

        QElapsedTimer timer;
        timer.start();
        aicore_cancel_scope_begin(m_cancelToken);
        char* json = aicore_rfdetr_detect_rgb_json(
                m_pendingCtx, rgb.constBits(), rgb.width(), rgb.height(),
                m_settings.threshold, m_settings.topK);
        aicore_cancel_scope_end(m_cancelToken);
        const double ms = static_cast<double>(timer.elapsed());

        if (!json) {
            const char* err = aicore_rfdetr_last_error(m_pendingCtx);
            emit logMessage(tr("[RF-DETR] Inference failed: %1")
                                    .arg(err ? QString::fromUtf8(err)
                                             : tr("unknown error")));
            return false;
        }

        RFDetrRunResult result;
        result.imagePath = m_settings.inputPath;
        result.imageName = QFileInfo(m_settings.inputPath).fileName();
        result.modelPath = m_settings.modelPath;
        result.runtimeMs = ms;
        result.resolvedDevice = m_settings.device;
        if (!RFDetrHelpers::parseDetectionsJson(QByteArray(json), &result)) {
            aicore_rfdetr_free_string(json);
            emit logMessage(tr("[RF-DETR] Failed to parse detection JSON."));
            return false;
        }
        aicore_rfdetr_free_string(json);

        // Fetch per-detection masks for segmentation models.
        if (aicore_rfdetr_context_has_segmentation(m_pendingCtx)) {
            const int n = aicore_rfdetr_detection_count(m_pendingCtx);
            for (int i = 0; i < n; ++i) {
                const int len = aicore_rfdetr_detection_mask_png(
                        m_pendingCtx, i, nullptr, 0);
                if (len <= 0) continue;
                QByteArray png;
                png.resize(len);
                if (aicore_rfdetr_detection_mask_png(
                            m_pendingCtx, i,
                            reinterpret_cast<unsigned char*>(png.data()),
                            len) == len) {
                    result.detections[i].maskPng = png;
                }
            }
        }

        // Annotated image (boxes + optional mask tint) for DB export.
        QImage annotated = rgb;
        RFDetrHelpers::drawDetections(&annotated, result.detections);
        result.annotatedImage = annotated;

        emit logMessage(
                tr("[RF-DETR] %1 object(s) in %2 ms (model=%3, threshold=%4)")
                        .arg(result.detections.size())
                        .arg(ms, 0, 'f', 1)
                        .arg(result.modelVariant)
                        .arg(m_settings.threshold, 0, 'f', 2));
        emit progressUpdate(1, 1);
        emit resultReady(result);
    }
    return true;
}
#endif
