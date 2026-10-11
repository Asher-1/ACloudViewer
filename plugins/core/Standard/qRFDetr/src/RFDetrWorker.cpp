// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RFDetrWorker.h"

#include <QtCompat.h>

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QImage>
#include <algorithm>

#ifdef AICore_ENABLED
#include "aicore/rfdetr_capi.h"
#include "aicore/runtime_capi.h"
#include "ecvAICoreRuntimeHelpers.h"
#endif

namespace {

aicore_image_format imageFormat(const QImage& image) {
    switch (image.format()) {
        case QImage::Format_RGB888:
            return AICORE_IMAGE_RGB8;
        case QImage::Format_RGBA8888:
            return AICORE_IMAGE_RGBA8;
        case QImage::Format_Grayscale8:
            return AICORE_IMAGE_GRAY8;
        case qtCompatQImageFormatBgr888():
            return AICORE_IMAGE_BGR8;
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

#ifdef AICore_ENABLED
// Device-task serialization moved to ecvAICoreRuntimeHelpers.h
// (ecvAICoreRuntime::makeDeviceTaskLock) — shared across all AICore plugins.
#endif

}  // namespace

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
    ecvAICoreRuntime::releasePending(m_pendingCtx, &aicore_rfdetr_free);
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
    auto taskGuard = ecvAICoreRuntime::makeDeviceTaskLock(m_settings.device);
    if (!taskGuard.isLocked()) {
        emit logMessage(
                tr("[RF-DETR] Failed to acquire the inference device; "
                   "another task is running."));
        return false;
    }
    // Warm up the backend on the UI thread is the caller's job; here we just
    // create the model context and run.
    aicore_rfdetr_options* opts = aicore_rfdetr_options_new();
    if (!opts) {
        emit logMessage(tr("[RF-DETR] Failed to allocate options."));
        return false;
    }
    aicore_rfdetr_options_set_device(opts,
                                     m_settings.device.toUtf8().constData());
    aicore_rfdetr_options_set_threads(opts, m_settings.threads);
    if (!m_settings.classFilter.isEmpty()) {
        aicore_rfdetr_options_set_class_filter(
                opts, m_settings.classFilter.constData(),
                static_cast<size_t>(m_settings.classFilter.size()));
        emit logMessage(tr("[RF-DETR] Class filter: %1 class(es) enabled")
                                .arg(m_settings.classFilter.size()));
    }

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

    emit logMessage(
            tr("[RF-DETR] Model loaded: variant=%1, classes=%2")
                    .arg(QString::fromUtf8(
                            aicore_rfdetr_context_variant(m_pendingCtx)))
                    .arg(aicore_rfdetr_context_num_classes(m_pendingCtx)));
    {
        // Model info envelope for the dialog's class-filter UI: variant,
        // class count and the full class-name table (indexed by class_id).
        emit modelInfoReady(RFDetrHelpers::modelInfoJsonFromCtx(m_pendingCtx));
    }
    emit progressUpdate(1, 1);

    if (aicore_cancel_token_requested(m_cancelToken)) {
        emit logMessage(tr("[RF-DETR] Cancelled before inference."));
        return false;
    }

    RFDetrRunResult result;
    QString error;
    if (!detectImage(m_pendingCtx, m_settings, m_cancelToken, &result,
                     &error)) {
        emit logMessage(tr("[RF-DETR] %1").arg(error));
        return false;
    }
    emit logMessage(
            tr("[RF-DETR] %1 object(s) in %2 ms (model=%3, threshold=%4)")
                    .arg(result.detections.size())
                    .arg(result.runtimeMs, 0, 'f', 1)
                    .arg(result.modelVariant)
                    .arg(m_settings.threshold, 0, 'f', 2));
    emit progressUpdate(1, 1);
    emit resultReady(result);
    return true;
}

bool RFDetrWorker::detectImage(aicore_rfdetr_ctx* ctx,
                               const Settings& settings,
                               aicore_cancel_token* cancel,
                               RFDetrRunResult* resultOut,
                               QString* error) {
    const QImage input(settings.inputPath);
    if (input.isNull()) {
        *error = tr("Failed to load image: %1").arg(settings.inputPath);
        return false;
    }
    const QImage rgb = inferenceImage(input);

    QElapsedTimer timer;
    timer.start();
    const aicore_image_view image{
            reinterpret_cast<const uint8_t*>(rgb.constBits()), rgb.width(),
            rgb.height(), static_cast<size_t>(rgb.bytesPerLine()),
            imageFormat(rgb)};
    if (cancel) aicore_cancel_scope_begin(cancel);
    const int detectRc = aicore_rfdetr_detect_image(
            ctx, &image, settings.threshold, settings.topK);
    if (cancel) aicore_cancel_scope_end(cancel);
    const double ms = static_cast<double>(timer.elapsed());

    if (detectRc != 0) {
        const char* err = aicore_rfdetr_last_error(ctx);
        *error = tr("Inference failed: %1")
                         .arg(err ? QString::fromUtf8(err)
                                  : tr("unknown error"));
        return false;
    }

    RFDetrRunResult& result = *resultOut;
    result.imagePath = settings.inputPath;
    result.imageName = QFileInfo(settings.inputPath).fileName();
    result.modelPath = settings.modelPath;
    result.runtimeMs = ms;
    // Backend-resolved device (may differ from the request when the GPU
    // lease failed and rfdetr fell back to CPU).
    const char* resolvedDevice = aicore_rfdetr_context_device(ctx);
    result.resolvedDevice = (resolvedDevice && resolvedDevice[0])
                                    ? QString::fromUtf8(resolvedDevice)
                                    : settings.device;
    result.modelVariant = QString::fromUtf8(aicore_rfdetr_context_variant(ctx));
    result.segmentation = aicore_rfdetr_context_has_segmentation(ctx) != 0;
    result.imageSize = static_cast<int>(aicore_rfdetr_context_image_size(ctx));
    result.numClasses =
            static_cast<int>(aicore_rfdetr_context_num_classes(ctx));
    const int detectionCount = aicore_rfdetr_detection_count(ctx);
    result.detections.reserve(detectionCount);
    for (int i = 0; i < detectionCount; ++i) {
        aicore_rfdetr_detection d{};
        if (aicore_rfdetr_detection_at(ctx, i, &d) != 0) continue;
        RFDetrDetection out;
        out.classId = d.class_id;
        out.className =
                d.class_name ? QString::fromUtf8(d.class_name) : QString();
        out.score = d.score;
        out.x1 = d.x1;
        out.y1 = d.y1;
        out.x2 = d.x2;
        out.y2 = d.y2;
        result.detections.append(out);
    }
    result.totalDetected = result.detections.size();
    result.resultJson.clear();

    // Fetch per-detection masks for segmentation models (raw bytes — no
    // PNG encode/decode round-trip; sizing also returns the dimensions).
    if (result.segmentation) {
        const int n = std::min(aicore_rfdetr_detection_count(ctx),
                               static_cast<int>(result.detections.size()));
        for (int i = 0; i < n; ++i) {
            int32_t mw = 0, mh = 0;
            const int len =
                    aicore_rfdetr_detection_mask(ctx, i, nullptr, 0, &mw, &mh);
            if (len <= 0) continue;
            QByteArray raw;
            raw.resize(len);
            if (aicore_rfdetr_detection_mask(
                        ctx, i, reinterpret_cast<unsigned char*>(raw.data()),
                        len, &mw, &mh) == len) {
                result.detections[i].maskRaw = raw;
                result.detections[i].maskWidth = mw;
                result.detections[i].maskHeight = mh;
            }
        }
    }

    // Annotated image (boxes + optional mask tint) for DB export.
    QImage annotated = rgb;
    RFDetrHelpers::drawDetections(&annotated, result.detections);
    result.annotatedImage = annotated;
    return true;
}
#endif
