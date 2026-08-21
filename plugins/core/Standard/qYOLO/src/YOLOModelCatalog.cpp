// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "YOLOModelCatalog.h"

#include <QColor>
#include <QFont>
#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPen>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "aicore/yolo_capi.h"

namespace YOLOHelpers {

QVector<YOLOModelEntry> catalogModels() {
    QVector<YOLOModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY);
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
        if (!e || !e->filename) continue;
        YOLOModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.task = QString::fromUtf8(e->task ? e->task : "detect");
        entry.depthCapable = e->depth_capable != 0;
        entry.end2end = e->end2end != 0;
        out.append(entry);
    }
#else
    (void)0;
#endif
    return out;
}

QVector<YOLOModelEntry> detectionModels() {
    return taskModels(QStringLiteral("detect"));
}

QVector<YOLOModelEntry> segmentModels() {
    return taskModels(QStringLiteral("segment"));
}

QVector<YOLOModelEntry> depthModels() {
    return taskModels(QStringLiteral("depth"));
}

QVector<YOLOModelEntry> taskModels(const QString& task) {
    QVector<YOLOModelEntry> out;
    const QVector<YOLOModelEntry> all = catalogModels();
    for (const YOLOModelEntry& e : all) {
        // taskModels filters on the GGUF task, so the pure-detect tab never
        // offers a segment model and vice versa; unknown tasks fall back to
        // the detect bucket for catalog forward-compatibility.
        if (e.task.isEmpty() ? task == QStringLiteral("detect")
                             : e.task == task) {
            out.append(e);
        }
    }
    return out;
}

bool findModelByFilename(const QString& filename, YOLOModelEntry* out) {
    const QVector<YOLOModelEntry> all = catalogModels();
    for (const YOLOModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_yolo_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_yolo_free_buffer(dir);
        return out;
    }
#else
    (void)0;
#endif
    return QString();
}

QString modelDisplayLabel(const YOLOModelEntry& entry) {
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

bool filenameIsDepth(const QString& filename) {
    return filename.toLower().contains(QStringLiteral("depth"));
}

bool parseDetectionsJson(const QByteArray& json, YOLORunResult* out) {
    if (out == nullptr) return false;
    out->detections.clear();
    out->resultJson = json;

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return false;

    const QJsonObject root = doc.object();
    out->modelVariant = root.value(QStringLiteral("model")).toString();
    out->end2end = root.value(QStringLiteral("end2end")).toInt() != 0;
    out->imageSize = root.value(QStringLiteral("image_size")).toInt();
    out->numClasses = root.value(QStringLiteral("num_classes")).toInt();

    const QJsonArray dets = root.value(QStringLiteral("detections")).toArray();
    for (const QJsonValue& v : dets) {
        if (!v.isObject()) continue;
        const QJsonObject d = v.toObject();
        YOLODetection det;
        det.classId = static_cast<uint32_t>(
                d.value(QStringLiteral("class_id")).toInt(0));
        det.className = d.value(QStringLiteral("class_name")).toString();
        det.score = static_cast<float>(
                d.value(QStringLiteral("score")).toDouble(0.0));
        const QJsonArray box = d.value(QStringLiteral("box")).toArray();
        if (box.size() == 4) {
            det.x1 = static_cast<float>(box.at(0).toDouble(0.0));
            det.y1 = static_cast<float>(box.at(1).toDouble(0.0));
            det.x2 = static_cast<float>(box.at(2).toDouble(0.0));
            det.y2 = static_cast<float>(box.at(3).toDouble(0.0));
        }
        out->detections.append(det);
    }
    out->totalDetected = out->detections.size();
    return true;
}

bool parseDepthStatsJson(const QByteArray& json, YOLODepthStats* out) {
    if (out == nullptr) return false;

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return false;

    const QJsonObject root = doc.object();
    out->width = root.value(QStringLiteral("depth_width")).toInt();
    out->height = root.value(QStringLiteral("depth_height")).toInt();
    out->minDepth = root.value(QStringLiteral("min_depth")).toDouble(0.0);
    out->maxDepth = root.value(QStringLiteral("max_depth")).toDouble(0.0);
    out->meanDepth = root.value(QStringLiteral("mean_depth")).toDouble(0.0);
    out->p95Depth = root.value(QStringLiteral("p95_depth")).toDouble(0.0);
    out->validPixels = static_cast<long long>(
            root.value(QStringLiteral("valid_pixels")).toDouble(0.0));
    return out->width > 0 && out->height > 0;
}

QRgb classColor(uint32_t classId) {
    // COCO-consistent deterministic palette (BGR order from OpenCV heritage).
    static const QRgb kPalette[20] = {
            qRgb(220, 20, 60),   qRgb(119, 11, 32),   qRgb(0, 0, 142),
            qRgb(0, 0, 230),     qRgb(106, 0, 228),   qRgb(0, 60, 100),
            qRgb(0, 80, 100),    qRgb(0, 0, 70),      qRgb(0, 0, 192),
            qRgb(250, 170, 30),  qRgb(100, 170, 30),  qRgb(220, 220, 0),
            qRgb(175, 116, 175), qRgb(250, 0, 30),    qRgb(165, 42, 42),
            qRgb(255, 77, 255),  qRgb(0, 226, 252),   qRgb(182, 182, 255),
            qRgb(0, 82, 0),      qRgb(120, 166, 157),
    };
    return kPalette[classId % 20];
}

namespace {

inline double turboChannel(double fourT, double offset) {
    return std::max(0.0, std::min(1.5 - std::fabs(fourT - offset), 1.0));
}

/** Shared turbo ramp: t in [0, 1] -> near (0) blue .. far (1) red. */
inline QRgb turboRgb(double t) {
    const double r = turboChannel(4.0 * t, 3.0);
    const double g = turboChannel(4.0 * t, 2.0);
    const double b = turboChannel(4.0 * t, 1.0);
    return qRgb(static_cast<int>(r * 255.0 + 0.5),
                static_cast<int>(g * 255.0 + 0.5),
                static_cast<int>(b * 255.0 + 0.5));
}

}  // namespace

void drawDetections(QImage* image,
                    const QVector<YOLODetection>& detections,
                    int thickness) {
    if (image == nullptr || image->isNull()) return;
    const int h = image->height();

    // Bind the painter to one stable ARGB32 data block for the whole call.
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }

    QPainter p(image);
    p.setRenderHint(QPainter::Antialiasing, false);

    QFont font = p.font();
    font.setPixelSize(std::max(12, h / 60));
    p.setFont(font);
    for (const YOLODetection& d : detections) {
        const QColor color(classColor(d.classId));
        QPen pen(color);
        pen.setWidth(thickness);
        p.setPen(pen);
        p.drawRect(QRectF(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));

        const QString label = QStringLiteral("%1 %2")
                                      .arg(d.className)
                                      .arg(d.score, 0, 'f', 2);
        // Anchor the banner above the box top, then keep it fully inside
        // the image: clamp horizontally, and flip below the box top when
        // the box hugs the top edge (the painter has no clipping here, so
        // off-canvas text would simply be invisible).
        QRect labelRect(static_cast<int>(d.x1),
                        static_cast<int>(d.y1) - font.pixelSize() - 6,
                        std::max(20, label.size() * font.pixelSize()),
                        font.pixelSize() + 6);
        labelRect.setWidth(
                std::min(labelRect.width(), std::max(20, image->width() - 4)));
        labelRect.moveLeft(std::clamp(
                labelRect.left(), 2,
                std::max(2, image->width() - labelRect.width() - 2)));
        if (labelRect.top() < 2) {
            labelRect.moveTop(static_cast<int>(d.y1) + 2);
        }
        labelRect.moveTop(std::min(
                labelRect.top(),
                std::max(2, image->height() - labelRect.height() - 2)));
        const QRect bg = labelRect.adjusted(0, 0, 4, 2);
        p.fillRect(bg.intersected(image->rect()), color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
        p.setPen(pen);
    }
    p.end();
}

void drawSegmentation(QImage* image,
                      const QVector<YOLOSegMask>& masks,
                      const QVector<YOLODetection>& detections,
                      int thickness) {
    if (image == nullptr || image->isNull() || masks.isEmpty()) return;
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }

    const int imgW = image->width();
    const int imgH = image->height();

    // Masks already live in the source-image space (AICore unscales them
    // from the letterbox canvas); a straight scale to the image keeps the
    // tint aligned with the boxes.
    for (int i = 0; i < masks.size(); ++i) {
        const YOLOSegMask& mask = masks[static_cast<size_t>(i)];
        if (mask.w <= 0 || mask.h <= 0 ||
            mask.bits.size() < static_cast<qint64>(mask.w) * mask.h) {
            continue;
        }
        // Grayscale view over a COPY of the mask bytes: QImage requires the
        // backing buffer to be 32-bit aligned, and QByteArray's offset is
        // not guaranteed to be (Qt 6 debug builds assert on misalignment).
        QImage maskImage(mask.w, mask.h, QImage::Format_Grayscale8);
        std::memcpy(maskImage.bits(), mask.bits.constData(),
                    static_cast<size_t>(mask.w) * mask.h);
        if (maskImage.isNull()) continue;
        maskImage = maskImage.scaled(imgW, imgH, Qt::IgnoreAspectRatio,
                                     Qt::FastTransformation);
        const QColor tint = i < detections.size()
                                    ? QColor(classColor(detections[i].classId))
                                    : QColor(220, 220, 220);

        // Alpha-blend the tint over the foreground mask pixels (no painter
        // needed — a straight per-pixel pass over the downscaled mask).
        for (int y = 0; y < imgH; ++y) {
            const uchar* src = maskImage.constScanLine(y);
            uchar* dst = image->scanLine(y);
            for (int x = 0; x < imgW; ++x) {
                if (src[x] == 0) continue;
                const int d = x * 4;
                dst[d] = static_cast<uchar>((dst[d] * 2 + tint.blue()) / 3);
                dst[d + 1] =
                        static_cast<uchar>((dst[d + 1] * 2 + tint.green()) / 3);
                dst[d + 2] =
                        static_cast<uchar>((dst[d + 2] * 2 + tint.red()) / 3);
            }
        }
    }

    drawDetections(image, detections, thickness);
}

QImage depthColorImage(const float* depth,
                       int width,
                       int height,
                       double minDepth,
                       double maxDepth) {
    if (depth == nullptr || width <= 0 || height <= 0) return QImage();
    const int n = width * height;

    double lo = minDepth;
    double hi = maxDepth;
    if (!(lo < hi)) {
        // Auto range over valid pixels (finite, > 0): min .. p95 — same rule
        // as the AICore depth statistics envelope.
        std::vector<float> valid;
        valid.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            const float d = depth[i];
            if (std::isfinite(d) && d > 0.0f) valid.push_back(d);
        }
        if (valid.empty()) return QImage();
        lo = *std::min_element(valid.begin(), valid.end());
        const size_t p95Idx =
                std::min((valid.size() * 95) / 100, valid.size() - 1);
        std::nth_element(valid.begin(), valid.begin() + p95Idx, valid.end());
        hi = valid[p95Idx];
    }
    if (!(lo < hi)) hi = lo + 1e-6;  // single-point range guard

    QImage out(width, height, QImage::Format_RGB888);
    const double invRange = 1.0 / (hi - lo);
    for (int y = 0; y < height; ++y) {
        uchar* row = out.scanLine(y);
        const float* src = depth + static_cast<size_t>(y) * width;
        for (int x = 0; x < width; ++x) {
            const float d = src[x];
            if (!(d > 0.0f) || !std::isfinite(d)) {
                // Invalid pixel (no depth): black.
                row[x * 3 + 0] = row[x * 3 + 1] = row[x * 3 + 2] = 0;
                continue;
            }
            double t = (d - lo) * invRange;
            t = std::max(0.0, std::min(1.0, t));
            const QRgb rgb = turboRgb(t);
            row[x * 3 + 0] = static_cast<uchar>(qRed(rgb));
            row[x * 3 + 1] = static_cast<uchar>(qGreen(rgb));
            row[x * 3 + 2] = static_cast<uchar>(qBlue(rgb));
        }
    }
    return out;
}

void drawDepthLegend(QImage* image, double minDepth, double maxDepth) {
    if (image == nullptr || image->isNull() || !(minDepth < maxDepth)) return;
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }

    QPainter p(image);
    const int w = image->width();
    const int h = image->height();
    const int barW = std::max(8, w / 60);
    const int barH = std::min(std::max(60, h / 2), 220);
    const int margin = 8;
    const int x0 = w - barW - margin;
    const int y0 = margin;

    // Ramp top = max (far, red) .. bottom = min (near, blue) — one mapping
    // shared with depthColorImage.
    for (int y = 0; y < barH; ++y) {
        const double t = 1.0 - static_cast<double>(y) / std::max(1, barH - 1);
        p.setPen(QPen(turboRgb(t)));
        p.drawLine(x0, y0 + y, x0 + barW - 1, y0 + y);
    }
    p.setPen(QPen(Qt::white, 1));
    p.drawRect(x0 - 1, y0 - 1, barW + 1, barH + 1);

    QFont font = p.font();
    font.setPixelSize(std::max(10, h / 70));
    p.setFont(font);
    p.setPen(Qt::white);
    const QString farLabel = QStringLiteral("%1 m").arg(maxDepth, 0, 'f', 1);
    const QString nearLabel = QStringLiteral("%1 m").arg(minDepth, 0, 'f', 1);
    const int labelWidth =
            std::max(farLabel.size(), nearLabel.size()) * font.pixelSize();
    p.drawText(QRect(x0 - labelWidth - 6, y0 - 2, labelWidth,
                     font.pixelSize() + 4),
               Qt::AlignRight | Qt::AlignVCenter, farLabel);
    p.drawText(QRect(x0 - labelWidth - 6, y0 + barH - font.pixelSize() - 2,
                     labelWidth, font.pixelSize() + 4),
               Qt::AlignRight | Qt::AlignVCenter, nearLabel);
    p.end();
}

}  // namespace YOLOHelpers
