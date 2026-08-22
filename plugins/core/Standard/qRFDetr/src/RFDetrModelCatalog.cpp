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
#include <algorithm>
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
        aicore_rfdetr_free_buffer(dir);
        return out;
    }
#else
    (void)0;
#endif
    return QString();
}

QString modelDisplayLabel(const RFDetrModelEntry& entry) {
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


/* 3-tap separable Gaussian blur [1,2,1]/4 on a Grayscale8 QImage.
 * Converts hard binary mask edges into a soft gradient so that
 * bilinear upscaling (SmoothPixmapTransform) produces smooth
 * boundaries instead of a staircase of per-pixel transitions. */
static void gaussianBlurMask3(QImage& img) {
    if (img.format() != QImage::Format_Grayscale8) return;
    const int w = img.width(), h = img.height();
    if (w <= 2 || h <= 2) return;
    QImage tmp(w, h, QImage::Format_Grayscale8);
    for (int y = 0; y < h; ++y) {
        const uchar* s = img.constScanLine(y);
        uchar* d = tmp.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const int l = (x > 0) ? s[x - 1] : 0;
            const int m = s[x];
            const int r = (x < w - 1) ? s[x + 1] : 0;
            d[x] = (uint8_t)((l + m * 2 + r) / 4);
        }
    }
    for (int y = 0; y < h; ++y) {
        uchar* d = img.scanLine(y);
        for (int x = 0; x < w; ++x) {
            const int t = (y > 0) ? tmp.constScanLine(y - 1)[x] : 0;
            const int m = tmp.constScanLine(y)[x];
            const int b = (y < h - 1) ? tmp.constScanLine(y + 1)[x] : 0;
            d[x] = (uint8_t)((t + m * 2 + b) / 4);
        }
    }
}

void drawDetections(QImage* image,
                    const QVector<RFDetrDetection>& detections,
                    float maskAlpha,
                    int thickness) {
    if (image == nullptr || image->isNull()) return;
    const int w = image->width();
    const int h = image->height();

    // Convert to packed ARGB32 before QPainter starts so the painter stays
    // bound to a stable data block for the whole call, and the mask tint
    // below can premultiply-blend through a single scaled blit per detection.
    if (image->format() != QImage::Format_ARGB32) {
        *image = image->convertToFormat(QImage::Format_ARGB32);
    }

    QPainter p(image);
    p.setRenderHint(QPainter::Antialiasing, false);
    // Smooth (bilinear) interpolation for the low-resolution mask → full-res
    // stretch so segmentation edges are anti-aliased instead of jagged.
    p.setRenderHint(QPainter::SmoothPixmapTransform, true);

    // Pass 1: tint masked regions (segmentation models).  All detections'
    // masks are accumulated into a single composite image at the mask
    // resolution (640x640) and then stretched to the frame with one
    // SIMD/hardware blit, instead of N separate drawImage calls (one per
    // detection).  Proportional alpha from the Gaussian-blurred mask
    // replaces the old binary threshold (>= 128 : tintRgb : 0) so the
    // blur's soft gradient is preserved through the bilinear upscale.
    QImage composite;
    for (const RFDetrDetection& d : detections) {
        if (d.maskRaw.isEmpty() || d.maskWidth <= 0 || d.maskHeight <= 0)
            continue;
        QImage mask(d.maskWidth, d.maskHeight, QImage::Format_Grayscale8);
        std::memcpy(mask.bits(), d.maskRaw.constData(),
                    static_cast<size_t>(d.maskWidth) * d.maskHeight);
        if (mask.isNull()) continue;
        gaussianBlurMask3(mask);

        if (composite.isNull()) {
            composite = QImage(mask.size(),
                               QImage::Format_ARGB32_Premultiplied);
            composite.fill(Qt::transparent);
        }
        const QRgb tintRgb = QColor(classColor(d.classId)).rgba();
        const QRgb tintPre = QColor(classColor(d.classId)).rgb();  // r,g,b only
        for (int y = 0; y < mask.height(); ++y) {
            const uchar* mrow = mask.constScanLine(y);
            QRgb* crow = reinterpret_cast<QRgb*>(composite.scanLine(y));
            for (int x = 0; x < mask.width(); ++x) {
                const int mv = mrow[x];
                if (mv <= 1) continue;
                if (mv > qAlpha(crow[x])) {
                    crow[x] = qRgba(
                            qRed(tintPre) * mv / 255,
                            qGreen(tintPre) * mv / 255,
                            qBlue(tintPre) * mv / 255, mv);
                }
            }
        }
    }
    if (!composite.isNull()) {
        p.setOpacity(maskAlpha);
        p.drawImage(QRect(0, 0, w, h), composite);
        p.setOpacity(1.0);
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

}  // namespace RFDetrHelpers
