// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RMBGWorker.h"

#include <QFileInfo>
#include <QImage>

#ifdef AICore_ENABLED
#include "aicore/rmbg_capi.h"
#include "aicore/runtime_capi.h"
#endif

RMBGWorker::RMBGWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
}

RMBGWorker::~RMBGWorker() {
    releaseContextOnMainThread();
#ifdef AICore_ENABLED
    if (m_cancelToken) {
        aicore_cancel_token_free(m_cancelToken);
        m_cancelToken = nullptr;
    }
#endif
}

void RMBGWorker::releaseContextOnMainThread() {
    // The context is created on the worker thread; destroy it here (main
    // thread) so GPU teardown never races the render thread.
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_rmbg_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

void RMBGWorker::requestTaskCancel() {
#ifdef AICore_ENABLED
    if (m_cancelToken) aicore_cancel_token_request(m_cancelToken);
#endif
}

void RMBGWorker::run() {
#ifdef AICore_ENABLED
    const bool ok = runInference();
    emit taskFinished(ok);
#else
    emit logMessage(tr("[RMBG] AICore is not enabled in this build."));
    emit taskFinished(false);
#endif
}

#ifdef AICore_ENABLED
bool RMBGWorker::runInference() {
    // Warm up the backend on the UI thread is the caller's job; here we just
    // create the model context and run.
    aicore_rmbg_options* opts = aicore_rmbg_options_new();
    if (!opts) {
        emit logMessage(tr("[RMBG] Failed to allocate options."));
        return false;
    }
    aicore_rmbg_options_set_device(opts,
                                   m_settings.device.toUtf8().constData());
    aicore_rmbg_options_set_threads(opts, m_settings.threads);

    emit logMessage(tr("[RMBG] Loading model: %1 (device=%2, threads=%3)")
                            .arg(QFileInfo(m_settings.modelPath).fileName(),
                                 m_settings.device)
                            .arg(m_settings.threads));
    emit progressUpdate(0, 1);

    m_pendingCtx = aicore_rmbg_load_opts(
            m_settings.modelPath.toUtf8().constData(), opts);
    aicore_rmbg_options_free(opts);
    if (!m_pendingCtx || !aicore_rmbg_is_ready(m_pendingCtx)) {
        const char* err = m_pendingCtx ? aicore_rmbg_last_error(m_pendingCtx)
                                       : "context allocation failed";
        emit logMessage(tr("[RMBG] Model load failed: %1")
                                .arg(err ? QString::fromUtf8(err)
                                         : tr("unknown error")));
        return false;
    }

    RMBGRunResult result;
    char* info = aicore_rmbg_info_json(m_pendingCtx);
    if (info) {
        const QString infoStr = QString::fromUtf8(info);
        emit modelInfoReady(infoStr);
        RMBGHelpers::parseInfoJson(QByteArray(info), &result);
        aicore_rmbg_free_string(info);
    }
    emit logMessage(tr("[RMBG] Model loaded: %1 (backend=%2, input=%3)")
                            .arg(result.modelVariant.isEmpty()
                                         ? QStringLiteral("RMBG-2.0")
                                         : result.modelVariant,
                                 result.backend)
                            .arg(result.inputSize));
    emit progressUpdate(1, 1);

    if (aicore_cancel_token_requested(m_cancelToken)) {
        emit logMessage(tr("[RMBG] Cancelled before inference."));
        return false;
    }

    // Single-image inference.
    {
        const QImage input(m_settings.inputPath);
        if (input.isNull()) {
            emit logMessage(tr("[RMBG] Failed to load image: %1")
                                    .arg(m_settings.inputPath));
            return false;
        }
        const QImage rgb = input.convertToFormat(QImage::Format_RGB888);
        emit progressUpdate(0, 1);

        QByteArray packedRgb;
        const uchar* rgbData = RMBGHelpers::packedRgb888Data(rgb, &packedRgb);
        if (!rgbData) {
            emit logMessage(tr("[RMBG] Failed to pack the RGB input."));
            return false;
        }
        aicore_cancel_scope_begin(m_cancelToken);
        uint8_t* rgba = nullptr;
        int32_t outW = 0, outH = 0;
        int rgbaLen = 0;
        const int rc = aicore_rmbg_remove_background_rgba(
                m_pendingCtx, rgbData, rgb.width(), rgb.height(), &rgba, &outW,
                &outH, &rgbaLen);
        aicore_cancel_scope_end(m_cancelToken);
        if (rc != 0 || !rgba || rgbaLen <= 0) {
            const char* err = aicore_rmbg_last_error(m_pendingCtx);
            emit logMessage(tr("[RMBG] Inference failed: %1")
                                    .arg(err ? QString::fromUtf8(err)
                                             : tr("unknown error")));
            if (rgba) aicore_rmbg_free_buffer(rgba);
            return false;
        }
        aicore_rmbg_timings timings{};
        if (aicore_rmbg_last_timings(m_pendingCtx, &timings) == 0) {
            result.preprocessMs = timings.preprocess_ms;
            result.runtimeMs = timings.inference_ms;
            result.postprocessMs = timings.postprocess_ms;
            result.totalRuntimeMs = timings.total_ms;
        }

        // Raw RGBA composite at original resolution — no PNG round-trip (a
        // PNG encode+decode round-trip was ~50 ms per frame). QImage takes
        // over the malloc'd API buffer via a free() cleanup function, so no
        // second full-frame copy; converting once to ARGB32 up front lets
        // thresholding and stats reuse the same buffer.
        QImage wrapped(
                rgba, outW, outH, outW * 4, QImage::Format_RGBA8888,
                [](void* info) { std::free(info); }, rgba);
        result.resultImage = wrapped.convertToFormat(QImage::Format_ARGB32);
        if (result.resultImage.isNull()) {
            emit logMessage(tr("[RMBG] Failed to wrap the result buffer."));
            return false;
        }

        if (m_settings.alphaThreshold > 0.0f) {
            result.resultImage = RMBGHelpers::applyAlphaThreshold(
                    result.resultImage, m_settings.alphaThreshold);
        }

        RMBGHelpers::computeAlphaStats(result.resultImage, &result.alphaMean,
                                       &result.foregroundRatio);
        result.imagePath = m_settings.inputPath;
        result.imageName = QFileInfo(m_settings.inputPath).fileName();
        result.modelPath = m_settings.modelPath;
        if (result.resolvedDevice.isEmpty()) {
            result.resolvedDevice = m_settings.device;
        }

        emit logMessage(
                tr("[RMBG] Removed background: graph %1 ms, total %2 ms (%3)")
                        .arg(result.runtimeMs, 0, 'f', 1)
                        .arg(result.totalRuntimeMs, 0, 'f', 1)
                        .arg(RMBGHelpers::formatAlphaStats(
                                result.alphaMean, result.foregroundRatio)));
        emit progressUpdate(1, 1);
        emit resultReady(result);
    }
    return true;
}
#endif
