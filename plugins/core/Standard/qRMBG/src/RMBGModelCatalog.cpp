// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RMBGModelCatalog.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>

#include <algorithm>
#include <cmath>

#include "aicore/rmbg_capi.h"

namespace RMBGHelpers {

QVector<RMBGModelEntry> catalogModels() {
    QVector<RMBGModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_rmbg_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_rmbg_model_entry* e = aicore_rmbg_model_at(i);
        if (!e || !e->filename) continue;
        RMBGModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        out.append(entry);
    }
#else
    (void)0;
#endif
    return out;
}

bool findModelByFilename(const QString& filename, RMBGModelEntry* out) {
    const QVector<RMBGModelEntry> all = catalogModels();
    for (const RMBGModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_rmbg_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_rmbg_free_string(dir);
        return out;
    }
#else
    (void)0;
#endif
    return QString();
}

bool parseInfoJson(const QByteArray& json, RMBGRunResult* out) {
    if (out == nullptr) return false;
    out->infoJson = json;

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return false;

    const QJsonObject root = doc.object();
    out->modelVariant = root.value(QStringLiteral("model")).toString();
    out->inputSize = root.value(QStringLiteral("input_size")).toInt();
    out->backend = root.value(QStringLiteral("backend")).toString();
    out->resolvedDevice = root.value(QStringLiteral("device")).toString();
    return true;
}

void computeAlphaStats(const QImage& rgba, double* alphaMean,
                       double* foregroundRatio) {
    if (alphaMean) *alphaMean = 0.0;
    if (foregroundRatio) *foregroundRatio = 0.0;
    if (rgba.isNull()) return;

    const QImage argb = rgba.convertToFormat(QImage::Format_ARGB32);
    const int w = argb.width();
    const int h = argb.height();
    if (w <= 0 || h <= 0) return;

    quint64 sum = 0;
    quint64 fg = 0;
    const qint64 total = static_cast<qint64>(w) * h;
    for (int y = 0; y < h; ++y) {
        const QRgb* row =
                reinterpret_cast<const QRgb*>(argb.constScanLine(y));
        for (int x = 0; x < w; ++x) {
            const int a = qAlpha(row[x]);
            sum += static_cast<quint64>(a);
            if (a >= 128) ++fg;
        }
    }
    if (total > 0) {
        if (alphaMean) {
            *alphaMean = static_cast<double>(sum) / (255.0 * total);
        }
        if (foregroundRatio) {
            *foregroundRatio = static_cast<double>(fg) / total;
        }
    }
}

QImage makeCheckerboard(const QSize& size, int cellSize) {
    if (size.isEmpty() || cellSize <= 0) return QImage();
    QImage img(size, QImage::Format_ARGB32);
    img.fill(Qt::white);
    const QRgb light = qRgb(235, 235, 235);
    const QRgb dark = qRgb(200, 200, 200);
    const int cols = (size.width() + cellSize - 1) / cellSize;
    const int rows = (size.height() + cellSize - 1) / cellSize;
    for (int cy = 0; cy < rows; ++cy) {
        for (int cx = 0; cx < cols; ++cx) {
            if (((cx + cy) & 1) == 0) continue;  // keep white
            const int x0 = cx * cellSize;
            const int y0 = cy * cellSize;
            const int x1 = std::min(x0 + cellSize, size.width());
            const int y1 = std::min(y0 + cellSize, size.height());
            for (int y = y0; y < y1; ++y) {
                QRgb* row = reinterpret_cast<QRgb*>(img.scanLine(y));
                for (int x = x0; x < x1; ++x) row[x] = dark;
            }
        }
    }
    Q_UNUSED(light);
    return img;
}

QImage compositeOnCheckerboard(const QImage& rgba, int cellSize) {
    if (rgba.isNull()) return QImage();
    const QImage argb = rgba.convertToFormat(QImage::Format_ARGB32);
    QImage out = makeCheckerboard(argb.size(), cellSize);
    if (out.isNull()) return QImage();
    QPainter p(&out);
    p.drawImage(0, 0, argb);
    p.end();
    return out;
}

QString formatAlphaStats(double alphaMean, double foregroundRatio) {
    return QStringLiteral("alpha %1%, fg %2%")
            .arg(alphaMean * 100.0, 0, 'f', 1)
            .arg(foregroundRatio * 100.0, 0, 'f', 1);
}

}  // namespace RMBGHelpers
