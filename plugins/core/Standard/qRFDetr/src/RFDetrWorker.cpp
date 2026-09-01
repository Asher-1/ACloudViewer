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

namespace {

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
    DeviceTaskGuard taskGuard(m_settings.device);
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

    // Single-image inference.
    {
        const QImage input(m_settings.inputPath);
        if (input.isNull()) {
            emit logMessage(tr("[RF-DETR] Failed to load image: %1")
                                    .arg(m_settings.inputPath));
            return false;
        }
        const QImage rgb = inferenceImage(input);
        emit progressUpdate(0, 1);

        QElapsedTimer timer;
        timer.start();
        const aicore_image_view image{
                reinterpret_cast<const uint8_t*>(rgb.constBits()), rgb.width(),
                rgb.height(), static_cast<size_t>(rgb.bytesPerLine()),
                imageFormat(rgb)};
        aicore_cancel_scope_begin(m_cancelToken);
        const int detectRc = aicore_rfdetr_detect_image(
                m_pendingCtx, &image, m_settings.threshold, m_settings.topK);
        aicore_cancel_scope_end(m_cancelToken);
        const double ms = static_cast<double>(timer.elapsed());

        if (detectRc != 0) {
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
        // Backend-resolved device (may differ from the request when the GPU
        // lease failed and rfdetr fell back to CPU).
        const char* resolvedDevice = aicore_rfdetr_context_device(m_pendingCtx);
        result.resolvedDevice = (resolvedDevice && resolvedDevice[0])
                                        ? QString::fromUtf8(resolvedDevice)
                                        : m_settings.device;
        result.modelVariant =
                QString::fromUtf8(aicore_rfdetr_context_variant(m_pendingCtx));
        result.segmentation =
                aicore_rfdetr_context_has_segmentation(m_pendingCtx) != 0;
        result.imageSize = static_cast<int>(
                aicore_rfdetr_context_image_size(m_pendingCtx));
        result.numClasses = static_cast<int>(
                aicore_rfdetr_context_num_classes(m_pendingCtx));
        const int detectionCount = aicore_rfdetr_detection_count(m_pendingCtx);
        result.detections.reserve(detectionCount);
        for (int i = 0; i < detectionCount; ++i) {
            aicore_rfdetr_detection d{};
            if (aicore_rfdetr_detection_at(m_pendingCtx, i, &d) != 0) continue;
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
        if (aicore_rfdetr_context_has_segmentation(m_pendingCtx)) {
            const int n = aicore_rfdetr_detection_count(m_pendingCtx);
            for (int i = 0; i < n; ++i) {
                int32_t mw = 0, mh = 0;
                const int len = aicore_rfdetr_detection_mask(
                        m_pendingCtx, i, nullptr, 0, &mw, &mh);
                if (len <= 0) continue;
                QByteArray raw;
                raw.resize(len);
                if (aicore_rfdetr_detection_mask(
                            m_pendingCtx, i,
                            reinterpret_cast<unsigned char*>(raw.data()), len,
                            &mw, &mh) == len) {
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
