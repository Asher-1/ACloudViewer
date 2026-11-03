// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VLXSeekWorker.h"

#include "VLXSeekModelCatalog.h"

#ifdef AICore_ENABLED
#include <aicore/image_view.h>
#include <aicore/inference_log.h>
#include <aicore/pipeline_timing.h>
#include <aicore/vlx_seek_capi.h>

#include <QtCompat.h>

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QPainter>
#include <QPen>

namespace {

// Qt-render the result in the official style: red width-3 rectangles plus
// the model-generated label at (xmin+2, ymin+2). Qt5/Qt6-safe drawing (no
// version-divergent APIs).
QImage annotate(const QImage& input, const VLXSeekRunResult& result) {
    QImage canvas = input.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&canvas);
    QPen pen(QColor(255, 0, 0), 3);
    painter.setPen(pen);
    const QFont font = painter.font();
    for (const VLXSeekRunResult::Box& box : result.boxes) {
        const QRectF r = box.rect;
        painter.drawRect(r);
        const QString text =
                box.label.isEmpty() ? QStringLiteral("?") : box.label;
        painter.drawText(QPointF(r.left() + 2,
                                 r.top() + 2 + QFontMetricsF(font).height()),
                         text);
    }
    painter.end();
    return canvas;
}

// QImage -> borrowed, stride-aware view over the caller's buffer. No format
// conversion here: the task converts RGB8/RGBA8/BGRA8/BGR8/GRAY8 natively,
// so the actual Qt buffer layout crosses unchanged (zero-copy hot path).
bool toImage(const QImage& image, aicore_image_view* view) {
    if (image.isNull()) return false;
    switch (image.format()) {
        case QImage::Format_RGB888:
            view->format = AICORE_IMAGE_RGB8;
            break;
        case QImage::Format_RGBA8888:
            view->format = AICORE_IMAGE_RGBA8;
            break;
        case qtCompatQImageFormatBgr888():
            view->format = AICORE_IMAGE_BGR8;
            break;
        case QImage::Format_Grayscale8:
            view->format = AICORE_IMAGE_GRAY8;
            break;
        default:
            return false;  // caller normalizes first
    }
    view->data = image.constBits();
    view->width = image.width();
    view->height = image.height();
    view->row_stride_bytes = static_cast<size_t>(image.bytesPerLine());
    return true;
}

}  // namespace

bool VLXSeekWorker::runImage(aicore_vlx_seek_ctx* ctx,
                             const Settings& settings,
                             VLXSeekRunResult* result,
                             QImage* annotated,
                             QString* error) {
    const QImage input(settings.inputPath);
    if (input.isNull()) {
        *error =
                QObject::tr("Failed to load image: %1").arg(settings.inputPath);
        return false;
    }
    // The task natively consumes RGB8/RGBA8/BGRA8/BGR8/GRAY8; normalize the
    // rare Qt formats it cannot borrow (indexed/ARGB32_PREMULTIPLIED/16-bit).
    QImage normalized = input;
    switch (input.format()) {
        case QImage::Format_RGB888:
        case QImage::Format_RGBA8888:
        case qtCompatQImageFormatBgr888():
        case QImage::Format_Grayscale8:
            break;
        default:
            normalized = input.convertToFormat(QImage::Format_RGB888);
            break;
    }

    aicore_image_view view{};
    if (!toImage(normalized, &view)) {
        *error = QObject::tr("Unsupported image format: %1")
                         .arg(normalized.format());
        return false;
    }

    QElapsedTimer timer;
    timer.start();
    // Region tasks (brief_region_caption / region_ocr) consume zero-based
    // proposal indexes; the runtime hard-rejects them without targets.
    QVector<int32_t> targets;
    if (!settings.targetIndexes.trimmed().isEmpty()) {
        targets = VLXSeekHelpers::parseTargetIndexes(settings.targetIndexes,
                                                     error);
        if (targets.isEmpty()) return false;
    }
    // External proposal injection (upstream --bbox-list parity): when a
    // boxes JSON is configured the built-in WeDetect-Uni proposer is
    // skipped and the given regions are verified directly.
    QVector<float> externalBoxes;
    if (!settings.externalBoxesPath.trimmed().isEmpty()) {
        externalBoxes = VLXSeekHelpers::parseExternalBoxes(
                settings.externalBoxesPath, error);
        if (externalBoxes.isEmpty()) return false;
    }
    const int rc = aicore_vlx_seek_run_image(
            ctx, &view, settings.task.toUtf8().constData(),
            settings.text.toUtf8().constData(),
            settings.lang.toUtf8().constData(),
            externalBoxes.isEmpty() ? nullptr : externalBoxes.constData(),
            static_cast<int32_t>(externalBoxes.size() / 4),
            targets.isEmpty() ? nullptr : targets.constData(),
            static_cast<int32_t>(targets.size()));
    const double ms = static_cast<double>(timer.elapsed());
    if (rc != 0) {
        const char* err = aicore_vlx_seek_last_error(ctx);
        *error = QObject::tr("Inference failed: %1")
                         .arg(err ? QString::fromUtf8(err)
                                  : QObject::tr("unknown error"));
        return false;
    }

    result->imagePath = settings.inputPath;
    result->answer = QString::fromUtf8(aicore_vlx_seek_result_answer(ctx));
    const int32_t count = aicore_vlx_seek_result_box_count(ctx);
    result->boxes.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
        float b[4] = {};
        aicore_vlx_seek_result_box(ctx, i, b);
        const char* label = aicore_vlx_seek_result_box_label(ctx, i);
        VLXSeekRunResult::Box box;
        box.rect = QRectF(b[0], b[1], b[2] - b[0], b[3] - b[1]);
        box.label = label ? QString::fromUtf8(label) : QString();
        result->boxes.push_back(box);
    }
    result->promptTokens = aicore_vlx_seek_result_prompt_tokens(ctx);
    result->completionTokens = aicore_vlx_seek_result_completion_tokens(ctx);
    result->proposalCount = aicore_vlx_seek_result_proposal_count(ctx);
    aicore_pipeline_timings t{};
    if (aicore_vlx_seek_last_pipeline_timings(ctx, &t) == 0) {
        result->preprocessMs = t.preprocess_ms;
        result->inferenceMs = t.inference_ms;
        result->postprocessMs = t.postprocess_ms;
        result->e2eMs = t.e2e_ms;
    }
    const char* device = aicore_vlx_seek_context_device(ctx);
    result->resolvedDevice = device ? QString::fromUtf8(device) : QString();
    const char* precision = aicore_vlx_seek_context_precision(ctx);
    result->precision = precision ? QString::fromUtf8(precision) : QString();
    if (annotated) *annotated = annotate(normalized, *result);
    return true;
}
#endif  // AICore_ENABLED

VLXSeekWorker::VLXSeekWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {}

VLXSeekWorker::~VLXSeekWorker() {
    // ~QThread aborts on a running thread; the pending context below must
    // outlive every run() access.
    if (isRunning()) {
        wait();
    }
    releaseContextOnMainThread();
}

void VLXSeekWorker::releaseContextOnMainThread() {
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_vlx_seek_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

#ifdef AICore_ENABLED
bool VLXSeekWorker::runInference() {
    // Context created and destroyed on the worker thread (AICore contract).
    aicore_vlx_seek_options* opts = aicore_vlx_seek_options_new();
    aicore_vlx_seek_options_set_device(opts,
                                       m_settings.device.toUtf8().constData());
    aicore_vlx_seek_options_set_threads(opts, m_settings.threads);
    aicore_vlx_seek_options_set_max_new_tokens(opts, m_settings.maxNewTokens);
    // The precision string only feeds introspection/metadata (the explicit
    // paths above are the loading truth); keep it truthful for model_info.
    if (!m_settings.precision.isEmpty())
        aicore_vlx_seek_options_set_precision(
                opts, m_settings.precision.toUtf8().constData());
    aicore_vlx_seek_options_set_models_dir(
            opts, QFileInfo(m_settings.visionPath).dir().path().toUtf8());
    aicore_vlx_seek_options_set_wedetect_gguf(
            opts, m_settings.wedetectPath.toUtf8().constData());
    aicore_vlx_seek_options_set_vision_gguf(
            opts, m_settings.visionPath.toUtf8().constData());
    aicore_vlx_seek_options_set_llm_gguf(
            opts, m_settings.llmPath.toUtf8().constData());
    aicore_vlx_seek_ctx* ctx = nullptr;
    {
        // Model-load wall time is a user-visible key fact (three GGUFs on
        // the first run dominate the perceived latency).
        QElapsedTimer loadTimer;
        loadTimer.start();
        ctx = aicore_vlx_seek_load_opts(opts);
        m_loadMs = double(loadTimer.elapsed());
    }
    aicore_vlx_seek_options_free(opts);

    // Log the task mode before inference so the console/log file identifies
    // which of the seven supported tasks is about to run (the "ready" line
    // above only identifies the device and precision, not the task).
    if (ctx && aicore_vlx_seek_is_ready(ctx)) {
        aicore_inference_log::log(
                QStringLiteral("[AICore vlx_seek] starting task=%1 | input=%2")
                        .arg(m_settings.task, m_settings.inputPath));
    }

    VLXSeekRunResult result;
    QImage annotated;
    QString error;
    const bool ok = ctx && aicore_vlx_seek_is_ready(ctx) &&
                    runImage(ctx, m_settings, &result, &annotated, &error);
    const char* loadErr =
            ctx && !ok ? aicore_vlx_seek_last_error(ctx) : nullptr;
    if (ctx) m_pendingCtx = ctx;  // freed on the main thread after the signal
    if (!ok) {
        if (loadErr && error.isEmpty()) error = QString::fromUtf8(loadErr);
        emit taskFinished(false);
        emit logMessage(error);
        return false;
    }
    result.annotatedImage = annotated;
    result.modelLoadMs = m_loadMs;
    emit resultReady(result);
    emit taskFinished(true);
    return true;
}
#endif  // AICore_ENABLED

void VLXSeekWorker::run() {
#ifdef AICore_ENABLED
    runInference();
#else
    emit taskFinished(false);
#endif
}
