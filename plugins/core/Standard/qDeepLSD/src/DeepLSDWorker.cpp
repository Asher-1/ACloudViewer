// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "DeepLSDWorker.h"

#include <QElapsedTimer>
#include <QFileInfo>
#include <QImage>
#include <algorithm>
#include <cstdlib>

#ifdef AICore_ENABLED
#include "aicore/deeplsd_capi.h"
#endif

DeepLSDWorker::DeepLSDWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    qRegisterMetaType<DeepLSDRunResult>("DeepLSDRunResult");
}

void DeepLSDWorker::releaseContextOnMainThread() {
#ifdef AICore_ENABLED
    if (m_pendingCtx) {
        aicore_deeplsd_free(m_pendingCtx);
        m_pendingCtx = nullptr;
    }
#endif
}

#ifdef AICore_ENABLED

namespace {

QImage make_distance_overlay(const QImage& gray,
                             const float* df,
                             int w,
                             int h) {
    QImage rgb = gray.convertToFormat(QImage::Format_RGB32);
    float maxv = 0.0f;
    const size_t n = static_cast<size_t>(w) * h;
    for (size_t i = 0; i < n; ++i) {
        maxv = std::max(maxv, df[i]);
    }
    if (maxv <= 1e-6f) {
        return rgb;
    }
    for (int y = 0; y < h; ++y) {
        auto* line = reinterpret_cast<QRgb*>(rgb.scanLine(y));
        for (int x = 0; x < w; ++x) {
            const float v = df[static_cast<size_t>(y) * w + x] / maxv;
            const int heat = static_cast<int>(255.0f * std::min(1.0f, v));
            const QRgb base = line[x];
            const int r = qMin(255, qRed(base) / 2 + heat);
            const int g = qGreen(base) / 2;
            const int b = qBlue(base) / 2;
            line[x] = qRgb(r, g, b);
        }
    }
    return rgb;
}

}  // namespace

bool DeepLSDWorker::runExtract() {
    if (m_settings.inputPath.isEmpty()) {
        emit logMessage("[Error] No input image.");
        return false;
    }

    QImage img(m_settings.inputPath);
    if (img.isNull()) {
        emit logMessage("[Error] Failed to load image.");
        return false;
    }
    QImage gray = img.convertToFormat(QImage::Format_Grayscale8);

    QElapsedTimer timer;
    timer.start();

    aicore_deeplsd_options* opts = aicore_deeplsd_options_new();
    if (!m_settings.device.isEmpty()) {
        aicore_deeplsd_options_set_device(
                opts, m_settings.device.toStdString().c_str());
    }
    aicore_deeplsd_options_set_threads(opts, m_settings.threads);

    aicore_deeplsd_ctx* ctx = aicore_deeplsd_load_opts(
            m_settings.modelPath.toStdString().c_str(), opts);
    aicore_deeplsd_options_free(opts);
    if (!ctx) {
        emit logMessage("[Error] Failed to create DeepLSD context.");
        return false;
    }
    if (const char* err = aicore_deeplsd_last_error(ctx)) {
        emit logMessage(QString("[Error] %1").arg(err));
        m_pendingCtx = ctx;
        return false;
    }

    emit progressUpdate(30, 100);
    float* df = nullptr;
    float* ang = nullptr;
    int32_t ow = 0;
    int32_t oh = 0;
    if (aicore_deeplsd_extract_gray(ctx, gray.constBits(), gray.width(),
                                    gray.height(), gray.bytesPerLine(), &df,
                                    &ang, &ow, &oh) != 0) {
        emit logMessage(
                QString("[Error] Extract failed: %1")
                        .arg(aicore_deeplsd_last_error(ctx) ?: "unknown"));
        m_pendingCtx = ctx;
        return false;
    }
    (void)ang;

    DeepLSDRunResult result;
    result.imagePath = m_settings.inputPath;
    result.imageName = QFileInfo(m_settings.inputPath).completeBaseName();
    result.width = ow;
    result.height = oh;
    result.overlay = make_distance_overlay(gray, df, ow, oh);
    result.runtimeMs = timer.elapsed();

    std::free(df);
    std::free(ang);
    m_pendingCtx = ctx;

    emit progressUpdate(100, 100);
    emit logMessage(QString("[DeepLSD] Extracted %1×%2 in %3 ms.")
                            .arg(ow)
                            .arg(oh)
                            .arg(result.runtimeMs, 0, 'f', 1));
    emit resultReady(result);
    return true;
}

#endif

void DeepLSDWorker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] AICore not enabled.");
    emit taskFinished(false);
#else
    const bool ok = runExtract();
    emit taskFinished(ok);
#endif
}
