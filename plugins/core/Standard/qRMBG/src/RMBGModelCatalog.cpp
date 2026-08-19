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
#include <cstring>

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

QString modelDisplayLabel(const RMBGModelEntry& entry) {
    QString label = entry.displayName;
    if (!entry.quantNote.isEmpty() && !label.contains(entry.quantNote)) {
        label += QStringLiteral(" ") + QChar(0x2014) + QStringLiteral(" ") +
                 entry.quantNote;
    }
    return label;
}

const uchar* packedRgb888Data(const QImage& image, QByteArray* scratch) {
    if (!scratch || image.isNull() || image.format() != QImage::Format_RGB888)
        return nullptr;
    scratch->clear();
    const int rowBytes = image.width() * 3;
    if (image.bytesPerLine() == rowBytes) return image.constBits();

    scratch->resize(rowBytes * image.height());
    for (int y = 0; y < image.height(); ++y) {
        std::memcpy(scratch->data() + y * rowBytes, image.constScanLine(y),
                    static_cast<size_t>(rowBytes));
    }
    return reinterpret_cast<const uchar*>(scratch->constData());
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
    out->mathProfile = root.value(QStringLiteral("math_profile")).toString();
    out->resolvedDevice = root.value(QStringLiteral("device")).toString();
    return true;
}

void computeAlphaStats(const QImage& rgba,
                       double* alphaMean,
                       double* foregroundRatio) {
    if (alphaMean) *alphaMean = 0.0;
    if (foregroundRatio) *foregroundRatio = 0.0;
    if (rgba.isNull()) return;

    /* The alpha byte sits at offset 3 of every pixel in both
     * Format_RGBA8888 (R,G,B,A bytes) and Format_ARGB32 (little-endian
     * 0xAARRGGBB), so a plain byte scan avoids a full-frame format
     * conversion. Other formats (no alpha byte, e.g. RGB888) fall back to
     * one RGBA8888 conversion. */
    QImage converted;
    const QImage* src = &rgba;
    if (rgba.format() != QImage::Format_ARGB32 &&
        rgba.format() != QImage::Format_RGBA8888) {
        converted = rgba.convertToFormat(QImage::Format_RGBA8888);
        src = &converted;
    }
    const int w = src->width();
    const int h = src->height();
    if (w <= 0 || h <= 0) return;

    quint64 sum = 0;
    quint64 fg = 0;
    const qint64 total = static_cast<qint64>(w) * h;
    for (int y = 0; y < h; ++y) {
        const uchar* row = src->constScanLine(y);
        for (int x = 0; x < w; ++x) {
            const int a = row[x * 4 + 3];
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

QImage applyAlphaThreshold(const QImage& rgba, float threshold) {
    if (rgba.isNull() || threshold <= 0.0f) return rgba;
    const float clamped = std::min(1.0f, std::max(0.0f, threshold));
    const int alphaCut = static_cast<int>(clamped * 255.0f + 0.5f);
    if (alphaCut <= 0) return rgba;
    if (alphaCut >= 255) return rgba;

    /* Same byte-offset trick as computeAlphaStats: alpha lives at offset 3
     * in both Format_ARGB32 (little-endian) and Format_RGBA8888, so we only
     * need a writable buffer of a compatible format — no pixel conversion
     * when the input already is one of them. detach() deep-copies only when
     * the buffer is still shared; the workers hand over an exclusively
     * owned image, so the hot path is a no-op (it used to copy() the whole
     * frame unconditionally). */
    QImage out;
    if (rgba.format() == QImage::Format_ARGB32 ||
        rgba.format() == QImage::Format_RGBA8888) {
        out = rgba;
        out.detach();
    } else {
        out = rgba.convertToFormat(QImage::Format_RGBA8888);
    }
    const int w = out.width();
    const int h = out.height();
    for (int y = 0; y < h; ++y) {
        uchar* row = out.scanLine(y);
        for (int x = 0; x < w; ++x) {
            /* Zero the alpha byte only — RGB untouched, matching the old
             * qRgba(r,g,b,0) rewrite byte-for-byte. */
            if (row[x * 4 + 3] < alphaCut) row[x * 4 + 3] = 0;
        }
    }
    return out;
}

QImage makeCheckerboard(const QSize& size, int cellSize) {
    if (size.isEmpty() || cellSize <= 0) return QImage();

    // The live video path composites every frame over this backdrop; a
    // 1080p pattern is 8 MB and rebuilding it per frame wasted ~3-6 ms.
    // Cache the last pattern (single-slot: the worker owns the calls, and
    // a size mismatch just rebuilds).
    static QImage cached;
    static QSize cachedSize;
    static int cachedCell = 0;
    if (!cached.isNull() && cached.size() == size && cachedCell == cellSize) {
        return cached;
    }

    QImage img(size, QImage::Format_ARGB32);
    img.fill(Qt::white);
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
    cached = img;
    cachedSize = size;
    cachedCell = cellSize;
    return cached;
}

QImage compositeOnCheckerboard(const QImage& rgba, int cellSize) {
    if (rgba.isNull()) return QImage();
    // ARGB32 in -> shared no-op conversion (the workers convert once up
    // front); other formats still get a one-shot conversion here.
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
