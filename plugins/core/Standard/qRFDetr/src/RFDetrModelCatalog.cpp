// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "RFDetrModelCatalog.h"

#include <QImage>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPainter>
#include <QPen>
#include <QThread>

#include <cmath>
#include <cstring>

#include "aicore/rfdetr_capi.h"

namespace RFDetrHelpers {

QVector<RFDetrModelEntry> catalogModels() {
    QVector<RFDetrModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_rfdetr_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_rfdetr_model_entry* e = aicore_rfdetr_model_at(i);
        if (!e || !e->filename) continue;
        RFDetrModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.segmentationCapable = e->segmentation_capable != 0;
        out.append(entry);
    }
#else
    (void)0;
#endif
    return out;
}

QVector<RFDetrModelEntry> detectionModels() {
    QVector<RFDetrModelEntry> out;
    const QVector<RFDetrModelEntry> all = catalogModels();
    for (const RFDetrModelEntry& e : all) {
        if (!e.segmentationCapable) out.append(e);
    }
    return out;
}

QVector<RFDetrModelEntry> segmentationModels() {
    QVector<RFDetrModelEntry> out;
    const QVector<RFDetrModelEntry> all = catalogModels();
    for (const RFDetrModelEntry& e : all) {
        if (e.segmentationCapable) out.append(e);
    }
    return out;
}

bool findModelByFilename(const QString& filename, RFDetrModelEntry* out) {
    const QVector<RFDetrModelEntry> all = catalogModels();
    for (const RFDetrModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_rfdetr_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_rfdetr_free_string(dir);
        return out;
    }
#else
    (void)0;
#endif
    return QString();
}

bool filenameIsSegmentation(const QString& filename) {
    const QString lower = filename.toLower();
    return lower.contains(QStringLiteral("seg"));
}

bool parseDetectionsJson(const QByteArray& json, RFDetrRunResult* out) {
    if (out == nullptr) return false;
    out->detections.clear();
    out->resultJson = json;

    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(json, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject()) return false;

    const QJsonObject root = doc.object();
    out->modelVariant = root.value(QStringLiteral("model")).toString();
    out->segmentation = root.value(QStringLiteral("segmentation")).toInt() != 0;
    out->imageSize = root.value(QStringLiteral("image_size")).toInt();
    out->numClasses = root.value(QStringLiteral("num_classes")).toInt();

    const QJsonArray dets = root.value(QStringLiteral("detections")).toArray();
    for (const QJsonValue& v : dets) {
        if (!v.isObject()) continue;
        const QJsonObject d = v.toObject();
        RFDetrDetection det;
        det.classId = static_cast<uint32_t>(
                d.value(QStringLiteral("class_id")).toInt(0));
        det.className = d.value(QStringLiteral("class_name")).toString();
        det.score = static_cast<float>(
                d.value(QStringLiteral("score")).toDouble(0.0));
        const QJsonArray box =
                d.value(QStringLiteral("box")).toArray();
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

QRgb classColor(uint32_t classId) {
    // COCO-consistent deterministic palette (BGR order from OpenCV heritage).
    static const QRgb kPalette[20] = {
            qRgb(220, 20, 60),    qRgb(119, 11, 32),   qRgb(0, 0, 142),
            qRgb(0, 0, 230),      qRgb(106, 0, 228),   qRgb(0, 60, 100),
            qRgb(0, 80, 100),     qRgb(0, 0, 70),      qRgb(0, 0, 192),
            qRgb(250, 170, 30),   qRgb(100, 170, 30),  qRgb(220, 220, 0),
            qRgb(175, 116, 175),  qRgb(250, 0, 30),    qRgb(165, 42, 42),
            qRgb(255, 77, 255),   qRgb(0, 226, 252),   qRgb(182, 182, 255),
            qRgb(0, 82, 0),       qRgb(120, 166, 157),
    };
    return kPalette[classId % 20];
}

void drawDetections(QImage* image,
                    const QVector<RFDetrDetection>& detections,
                    float maskAlpha,
                    int thickness) {
    if (image == nullptr || image->isNull()) return;
    const int w = image->width();
    const int h = image->height();

    QPainter p(image);
    p.setRenderHint(QPainter::Antialiasing, false);

    // Pass 1: tint masked regions (segmentation models).
    for (const RFDetrDetection& d : detections) {
        if (d.maskPng.isEmpty()) continue;
        const QImage mask = QImage::fromData(d.maskPng, "PNG");
        if (mask.isNull() || mask.width() != w || mask.height() != h) continue;
        const QColor tint(classColor(d.classId));
        for (int y = 0; y < h; ++y) {
            const uchar* row = mask.constScanLine(y);
            for (int x = 0; x < w; ++x) {
                if (row[x] < 128) continue;
                const QRgb px = image->pixel(x, y);
                image->setPixel(
                        x, y,
                        qRgb(static_cast<int>(tint.red() * maskAlpha +
                                              qRed(px) * (1.0 - maskAlpha)),
                             static_cast<int>(tint.green() * maskAlpha +
                                              qGreen(px) * (1.0 - maskAlpha)),
                             static_cast<int>(tint.blue() * maskAlpha +
                                              qBlue(px) * (1.0 - maskAlpha))));
            }
        }
    }

    // Pass 2: boxes + labels.
    QFont font = p.font();
    font.setPixelSize(std::max(12, h / 60));
    p.setFont(font);
    for (const RFDetrDetection& d : detections) {
        const QColor color(classColor(d.classId));
        QPen pen(color);
        pen.setWidth(thickness);
        p.setPen(pen);
        p.drawRect(QRectF(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1));

        const QString label = QStringLiteral("%1 %.2f")
                                      .arg(d.className)
                                      .arg(d.score, 0, 'f', 2);
        const QRect labelRect(static_cast<int>(d.x1),
                              static_cast<int>(d.y1) - font.pixelSize() - 6,
                              std::max(20, label.size() * font.pixelSize()),
                              font.pixelSize() + 6);
        const QRect bg = labelRect.adjusted(0, 0, 4, 2);
        p.fillRect(bg.intersected(image->rect()), color);
        p.setPen(Qt::white);
        p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
        p.setPen(pen);
    }
    p.end();
}

}  // namespace RFDetrHelpers
