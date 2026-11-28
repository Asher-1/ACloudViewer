// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "WeVisOneWorker.h"

#include "WeVisOneModelCatalog.h"

#ifdef AICore_ENABLED
#include <QtCompat.h>
#include <aicore/image_view.h>
#include <aicore/pipeline_timing.h>
#include <aicore/wevisone_capi.h>

#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <algorithm>

namespace {

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

bool WeVisOneWorker::runImage(aicore_wevisone_ctx* ctx,
                              const Settings& settings,
                              WeVisOneRunResult* result,
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

    // visual_prompt consumes reference boxes (the <ref>/<box> tokens);
    // every other task ignores them.
    QVector<float> refBoxes;
    if (settings.task == QStringLiteral("visual_prompt")) {
        refBoxes = WeVisOneHelpers::parseRefBoxes(settings.refBoxes, error);
        if (refBoxes.isEmpty()) {
            if (error->isEmpty()) {
                *error = QObject::tr(
                        "Task visual_prompt requires reference boxes "
                        "(x1,y1,x2,y2 in image pixels).");
            }
            return false;
        }
    }

    const int rc = aicore_wevisone_run_image(
            ctx, &view, settings.task.toUtf8().constData(),
            settings.query.toUtf8().constData(),
            refBoxes.isEmpty() ? nullptr : refBoxes.constData(),
            static_cast<int32_t>(refBoxes.size() / 4));
    if (rc != 0) {
        const char* err = aicore_wevisone_last_error(ctx);
        *error = QObject::tr("Inference failed: %1")
                         .arg(err ? QString::fromUtf8(err)
                                  : QObject::tr("unknown error"));
        return false;
    }

    result->imagePath = settings.inputPath;
    result->answer = QString::fromUtf8(aicore_wevisone_result_answer(ctx));
    const int32_t count = aicore_wevisone_result_box_count(ctx);
    result->boxes.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
        float b[4] = {};
        aicore_wevisone_result_box(ctx, i, b);
        result->boxes.push_back(QRectF(b[0], b[1], b[2] - b[0], b[3] - b[1]));
    }
    result->promptTokens = aicore_wevisone_result_prompt_tokens(ctx);
    result->completionTokens = aicore_wevisone_result_completion_tokens(ctx);
    result->hasMasks = aicore_wevisone_result_has_masks(ctx) != 0;
    result->matteCount = aicore_wevisone_result_matte_count(ctx);
    if (result->hasMasks && result->matteCount > 0) {
        // Only matte 0 is retained by the upstream RunResult contract;
        // copy it out before the next run call invalidates the borrow.
        int32_t mw = 0, mh = 0;
        if (aicore_wevisone_result_matte_size(ctx, 0, &mw, &mh) == 0) {
            const float* data = aicore_wevisone_result_matte_data(ctx, 0);
            if (data && mw > 0 && mh > 0) {
                result->matteWidth = mw;
                result->matteHeight = mh;
                const size_t n = static_cast<size_t>(mw) * mh;
                result->matte.resize(static_cast<int>(n));
                std::copy(data, data + n, result->matte.data());
            }
        }
    }
    aicore_pipeline_timings t{};
    if (aicore_wevisone_last_pipeline_timings(ctx, &t) == 0) {
        result->preprocessMs = t.preprocess_ms;
        result->inferenceMs = t.inference_ms;
        result->postprocessMs = t.postprocess_ms;
        result->e2eMs = t.e2e_ms;
    }
    const char* device = aicore_wevisone_context_device(ctx);
    result->resolvedDevice = device ? QString::fromUtf8(device) : QString();
    const char* precision = aicore_wevisone_context_precision(ctx);
    result->precision = precision ? QString::fromUtf8(precision) : QString();
    if (annotated) {
        *annotated = WeVisOneHelpers::annotateResult(
                normalized, result->boxes,
                result->hasMasks ? result->matte : QVector<float>(),
                result->matteWidth, result->matteHeight);
    }
    return true;
}
#endif  // AICore_ENABLED

WeVisOneWorker::WeVisOneWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {}

WeVisOneWorker::~WeVisOneWorker() {
    // ~QThread aborts on a running thread; the pending context below must
    // outlive every run() access.
    if (isRunning()) {
        wait();
    }
    releaseContextOnMainThread();
}

void WeVisOneWorker::releaseContextOnMainThread() {
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_wevisone_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

#ifdef AICore_ENABLED
bool WeVisOneWorker::runInference() {
    // Context created and destroyed on the worker thread (AICore contract).
    aicore_wevisone_options* opts = aicore_wevisone_options_new();
    aicore_wevisone_options_set_device(opts,
                                       m_settings.device.toUtf8().constData());
    aicore_wevisone_options_set_threads(opts, m_settings.threads);
    aicore_wevisone_options_set_max_new_tokens(opts, m_settings.maxNewTokens);
    aicore_wevisone_options_set_decode_masks(opts,
                                             m_settings.decodeMasks ? 1 : 0);
    // The size/precision strings only feed introspection/metadata (the
    // explicit paths below are the loading truth); keep them truthful.
    aicore_wevisone_options_set_size(opts,
                                     m_settings.size.toUtf8().constData());
    aicore_wevisone_options_set_precision(
            opts, m_settings.precision.toUtf8().constData());
    aicore_wevisone_options_set_decoder_precision(
            opts, m_settings.decoderPrecision.toUtf8().constData());
    aicore_wevisone_options_set_models_dir(
            opts, QFileInfo(m_settings.vlmPath).dir().path().toUtf8());
    aicore_wevisone_options_set_vlm_gguf(
            opts, m_settings.vlmPath.toUtf8().constData());
    aicore_wevisone_options_set_mmproj_gguf(
            opts, m_settings.mmprojPath.toUtf8().constData());
    aicore_wevisone_options_set_decoder_gguf(
            opts, m_settings.decoderPath.toUtf8().constData());
    QElapsedTimer loadTimer;
    loadTimer.start();
    aicore_wevisone_ctx* ctx = aicore_wevisone_load_opts(opts);
    aicore_wevisone_options_free(opts);
    const double modelLoadMs = static_cast<double>(loadTimer.elapsed());

    WeVisOneRunResult result;
    result.modelLoadMs = modelLoadMs;
    QImage annotated;
    QString error;
    const bool ok = ctx && aicore_wevisone_is_ready(ctx) &&
                    runImage(ctx, m_settings, &result, &annotated, &error);
    const char* loadErr =
            ctx && !ok ? aicore_wevisone_last_error(ctx) : nullptr;
    if (ctx) m_pendingCtx = ctx;  // freed on the main thread after the signal
    if (!ok) {
        if (loadErr && error.isEmpty()) error = QString::fromUtf8(loadErr);
        emit taskFinished(false);
        emit logMessage(error);
        return false;
    }
    result.annotatedImage = annotated;
    emit resultReady(result);
    emit taskFinished(true);
    return true;
}
#endif  // AICore_ENABLED

void WeVisOneWorker::run() {
#ifdef AICore_ENABLED
    runInference();
#else
    emit taskFinished(false);
#endif
}