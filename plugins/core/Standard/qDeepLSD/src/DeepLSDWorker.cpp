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
#include <QImageReader>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPen>
#include <algorithm>
#include <cstdlib>

#ifdef AICore_ENABLED
#include "aicore/backend_capi.h"
#include "aicore/deeplsd_capi.h"
#endif
#include "aicore/inference_log.h"
#include "aicore/runtime_capi.h"

DeepLSDWorker::DeepLSDWorker(const Settings& settings, QObject* parent)
    : QThread(parent), m_settings(settings) {
    qRegisterMetaType<DeepLSDRunResult>("DeepLSDRunResult");
#ifdef AICore_ENABLED
    m_cancelToken = aicore_cancel_token_new();
#endif
}

DeepLSDWorker::~DeepLSDWorker() {
#ifdef AICore_ENABLED
    aicore_cancel_token_free(m_cancelToken);
#endif
}

void DeepLSDWorker::requestTaskCancel() {
    requestInterruption();
#ifdef AICore_ENABLED
    aicore_cancel_token_request(m_cancelToken);
#endif
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

constexpr int kDefaultMaxResize = 1024;

struct LoadedGray {
    QImage original;
    QImage processed;
    double scaleX = 1.0;
    double scaleY = 1.0;
};

LoadedGray load_gray_resized(const QString& path, int max_resize) {
    QImageReader reader(path);
    reader.setAutoTransform(true);
    const QImage img = reader.read();
    if (img.isNull()) {
        return {};
    }
    LoadedGray out;
    out.original = img.convertToFormat(QImage::Format_Grayscale8);
    out.processed = out.original;
    if (max_resize > 0) {
        const int max_dim =
                std::max(out.original.width(), out.original.height());
        if (max_dim > max_resize) {
            out.processed = out.original.scaled(max_resize, max_resize,
                                                Qt::KeepAspectRatio,
                                                Qt::SmoothTransformation);
            out.scaleX = static_cast<double>(out.original.width()) /
                         out.processed.width();
            out.scaleY = static_cast<double>(out.original.height()) /
                         out.processed.height();
        }
    }
    return out;
}

DeepLSDLineSegment scale_segment(const aicore_deeplsd_segment& seg,
                                 double scaleX,
                                 double scaleY) {
    DeepLSDLineSegment out;
    out.x1 = static_cast<float>(seg.x1 * scaleX);
    out.y1 = static_cast<float>(seg.y1 * scaleY);
    out.x2 = static_cast<float>(seg.x2 * scaleX);
    out.y2 = static_cast<float>(seg.y2 * scaleY);
    out.score = seg.score;
    return out;
}

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

QImage make_line_visualization(
        const QImage& gray, const std::vector<DeepLSDLineSegment>& segments) {
    QImage rgb = gray.convertToFormat(QImage::Format_RGB32);
    QPainter painter(&rgb);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.setPen(QPen(QColor(DeepLSDLineStyle::kRed, DeepLSDLineStyle::kGreen,
                               DeepLSDLineStyle::kBlue),
                        1));
    for (const DeepLSDLineSegment& seg : segments) {
        painter.drawLine(QPointF(seg.x1, seg.y1), QPointF(seg.x2, seg.y2));
    }
    return rgb;
}

}  // namespace

bool DeepLSDWorker::runExtract() {
    if (m_settings.inputPath.isEmpty()) {
        emit logMessage("[Error] No input image.");
        return false;
    }

    const LoadedGray loaded =
            load_gray_resized(m_settings.inputPath, kDefaultMaxResize);
    if (loaded.processed.isNull()) {
        emit logMessage("[Error] Failed to load image.");
        return false;
    }
    emit logMessage(QString("[DeepLSD] Processing %1×%2 (source %3×%4)")
                            .arg(loaded.processed.width())
                            .arg(loaded.processed.height())
                            .arg(loaded.original.width())
                            .arg(loaded.original.height()));

    QElapsedTimer timer;
    timer.start();

    aicore_inference_log::log_device_request(QStringLiteral("DeepLSD"),
                                             m_settings.device);
    emit logMessage("[DeepLSD] Loading model: " + m_settings.modelPath);

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
    if (char* info = aicore_deeplsd_info_json(ctx)) {
        const QJsonObject obj =
                QJsonDocument::fromJson(QByteArray(info)).object();
        aicore_deeplsd_free_string(info);
        const QString resolved = obj.value(QStringLiteral("device")).toString();
        aicore_inference_log::log_device_resolved(QStringLiteral("DeepLSD"),
                                                  resolved);
    }

    emit progressUpdate(30, 100);
    float* df = nullptr;
    float* ang = nullptr;
    aicore_deeplsd_segment* segs = nullptr;
    int32_t seg_count = 0;
    int32_t ow = 0;
    int32_t oh = 0;
    if (aicore_deeplsd_extract_segments(
                ctx, loaded.processed.constBits(), loaded.processed.width(),
                loaded.processed.height(), loaded.processed.bytesPerLine(),
                &segs, &seg_count, &df, &ang, &ow, &oh) != 0) {
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
    result.originalWidth = loaded.original.width();
    result.originalHeight = loaded.original.height();
    result.segments.reserve(static_cast<size_t>(seg_count));
    for (int32_t i = 0; i < seg_count; ++i) {
        DeepLSDLineSegment seg =
                scale_segment(segs[i], loaded.scaleX, loaded.scaleY);
        if (m_settings.minSegmentScore > 0.0f &&
            seg.score < m_settings.minSegmentScore) {
            continue;
        }
        result.segments.push_back(seg);
    }
    if (char* info = aicore_deeplsd_info_json(ctx)) {
        const QJsonObject obj =
                QJsonDocument::fromJson(QByteArray(info)).object();
        aicore_deeplsd_free_string(info);
        result.resolvedDevice = obj.value(QStringLiteral("device")).toString();
    }
    result.lineVisualization =
            make_line_visualization(loaded.original, result.segments);
    if (m_settings.computeDistanceOverlay) {
        QImage procOverlay =
                make_distance_overlay(loaded.processed, df, ow, oh);
        result.distanceOverlay = procOverlay.scaled(loaded.original.size(),
                                                    Qt::IgnoreAspectRatio,
                                                    Qt::SmoothTransformation);
    }
    result.runtimeMs = timer.elapsed();

    std::free(df);
    std::free(ang);
    std::free(segs);
    m_pendingCtx = ctx;

    emit progressUpdate(100, 100);
    aicore_inference_log::log_inference_done(
            QStringLiteral("DeepLSD"), result.resolvedDevice, result.runtimeMs,
            QStringLiteral("%1 segments on %2×%3")
                    .arg(result.segments.size())
                    .arg(result.originalWidth)
                    .arg(result.originalHeight));
    emit resultReady(result);
    return true;
}

#endif

void DeepLSDWorker::run() {
#ifndef AICore_ENABLED
    emit logMessage("[Error] AICore not enabled.");
    emit taskFinished(false);
#else
    aicore_device_task_lock(m_settings.device.toUtf8().constData());
    aicore_cancel_scope_begin(m_cancelToken);
    const bool ok = runExtract();
    aicore_cancel_scope_end(m_cancelToken);
    aicore_device_task_unlock();
    emit taskFinished(ok);
#endif
}
