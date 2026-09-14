// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "GKDModelCatalog.h"

#include <QPainter>
#include <QRegularExpression>
#include <algorithm>
#include <cmath>
#include <cstring>

#include "aicore/gkd_capi.h"

namespace GKDHelpers {

namespace {

// Role-filtered catalog view over the AICore GKD catalog.
QVector<GKDModelEntry> catalogFromApi() {
    QVector<GKDModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_gkd_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_gkd_model_entry* e = aicore_gkd_model_at(i);
        if (!e || !e->filename) continue;
        GKDModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.sizeBytes = static_cast<qint64>(e->size_bytes);
        out.append(entry);
    }
#endif
    return out;
}

}  // namespace

QVector<GKDModelEntry> catalogModels() { return catalogFromApi(); }

int catalogDefaultIndex() {
#ifdef AICore_ENABLED
    return aicore_gkd_model_default_index();
#else
    return -1;
#endif
}

bool findModelByFilename(const QString& filename, GKDModelEntry* out) {
    const QVector<GKDModelEntry> all = catalogModels();
    for (const GKDModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_gkd_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_gkd_free_buffer(dir);
        return out;
    }
#endif
    return QString();
}

QString modelDisplayLabel(const GKDModelEntry& entry) {
    QString label = entry.displayName;
    if (!entry.quantNote.isEmpty() && !label.contains(entry.quantNote)) {
        label += QStringLiteral(" ") + QChar(0x2014) + QStringLiteral(" ") +
                 entry.quantNote;
    }
    return label;
}

bool imageView(QImage* image, aicore_image_view* out) {
    if (image == nullptr || out == nullptr || image->isNull()) return false;
    // Keep the borrowed view zero-copy for the formats AICore understands.
    switch (image->format()) {
        case QImage::Format_RGB888:
            break;  // RGB8, zero-copy
        case QImage::Format_Grayscale8:
            break;  // GRAY8, zero-copy
        case QImage::Format_ARGB32:
            // Little-endian ARGB32 memory layout is BGRA8.
#if Q_BYTE_ORDER == Q_LITTLE_ENDIAN
            break;
#else
            *image = image->convertToFormat(QImage::Format_RGBA8888);
            break;
#endif
        case QImage::Format_RGBA8888:
        case QImage::Format_BGR888:
            break;
        default:
            *image = image->convertToFormat(QImage::Format_ARGB32);
            if (image->format() != QImage::Format_ARGB32) return false;
#if Q_BYTE_ORDER != Q_LITTLE_ENDIAN
            *image = image->convertToFormat(QImage::Format_RGBA8888);
#endif
            break;
    }
    if (image->isNull()) return false;
    out->data = image->constBits();
    out->width = image->width();
    out->height = image->height();
    out->row_stride_bytes = static_cast<size_t>(image->bytesPerLine());
    switch (image->format()) {
        case QImage::Format_RGB888:
            out->format = AICORE_IMAGE_RGB8;
            break;
        case QImage::Format_Grayscale8:
            out->format = AICORE_IMAGE_GRAY8;
            break;
        case QImage::Format_ARGB32:
            out->format = AICORE_IMAGE_BGRA8;
            break;
        case QImage::Format_RGBA8888:
            out->format = AICORE_IMAGE_RGBA8;
            break;
        case QImage::Format_BGR888:
            out->format = AICORE_IMAGE_BGR8;
            break;
        default:
            return false;
    }
    return out->data != nullptr && out->width > 0 && out->height > 0;
}

bool parseCoordinatePairs(const QString& text, QVector<QPointF>* out) {
    if (out == nullptr) return false;
    out->clear();
    const QString trimmed = text.simplified();
    if (trimmed.isEmpty()) return true;
    static const QRegularExpression number(
            QStringLiteral("-?\\d+(?:\\.\\d+)?"));
    QRegularExpressionMatchIterator it = number.globalMatch(trimmed);
    QVector<double> values;
    QVector<QPair<int, int>> spans;
    while (it.hasNext()) {
        const auto m = it.next();
        values.append(m.captured(0).toDouble());
        spans.append({m.capturedStart(), m.capturedLength()});
    }
    if (values.size() < 2 || values.size() % 2 != 0) return false;
    // Reject leftover garbage: every character outside a number must be
    // whitespace or the pair separator (',').
    for (int i = 0; i < trimmed.size(); ++i) {
        const QChar ch = trimmed.at(i);
        if (ch.isSpace() || ch == QLatin1Char(',')) continue;
        bool insideNumber = false;
        for (const auto& span : spans) {
            if (i >= span.first && i < span.first + span.second) {
                insideNumber = true;
                break;
            }
        }
        if (!insideNumber) return false;
    }
    out->reserve(values.size() / 2);
    for (int i = 0; i + 1 < values.size(); i += 2) {
        out->append(QPointF(values[i], values[i + 1]));
    }
    return true;
}

QStringList splitPrompts(const QString& text) {
    QStringList prompts;
    QString current;
    bool inQuotes = false;
    for (const QChar ch : text) {
        if (ch == QLatin1Char('"')) {
            inQuotes = !inQuotes;
        } else if (ch == QLatin1Char(',') && !inQuotes) {
            const QString trimmed = current.trimmed();
            if (!trimmed.isEmpty()) prompts.append(trimmed);
            current.clear();
        } else {
            current.append(ch);
        }
    }
    const QString trimmed = current.trimmed();
    if (!trimmed.isEmpty()) prompts.append(trimmed);
    return prompts;
}

QColor groupColor(int groupId) {
    // COCO-consistent deterministic palette (same heritage as qYOLO).
    static const QRgb kPalette[20] = {
            qRgb(220, 20, 60),   qRgb(119, 11, 32),   qRgb(0, 0, 142),
            qRgb(0, 0, 230),     qRgb(106, 0, 228),   qRgb(0, 60, 100),
            qRgb(0, 80, 100),    qRgb(0, 0, 70),      qRgb(0, 0, 192),
            qRgb(250, 170, 30),  qRgb(100, 170, 30),  qRgb(220, 220, 0),
            qRgb(175, 116, 175), qRgb(250, 0, 30),    qRgb(165, 42, 42),
            qRgb(255, 77, 255),  qRgb(0, 226, 252),   qRgb(182, 182, 255),
            qRgb(0, 82, 0),      qRgb(120, 166, 157),
    };
    return QColor(kPalette[groupId % 20]);
}

QImage renderResult(const QImage& source,
                    const QVector<GKDKeypointSet>& sets,
                    float minScore) {
    if (source.isNull()) return QImage();
    QImage out = source.format() == QImage::Format_ARGB32
                         ? source
                         : source.convertToFormat(QImage::Format_ARGB32);

    QPainter p(&out);
    p.setRenderHint(QPainter::Antialiasing, true);

    QFont font = p.font();
    font.setPixelSize(std::max(12, out.height() / 50));
    p.setFont(font);

    int groupId = 0;
    for (const GKDKeypointSet& set : sets) {
        const QColor color = groupColor(groupId++);
        if (set.hasBox) {
            QPen pen(color);
            pen.setWidth(2);
            p.setPen(pen);
            p.setBrush(Qt::NoBrush);
            p.drawRect(
                    QRectF(set.x1, set.y1, set.x2 - set.x1, set.y2 - set.y1));
            if (!set.label.isEmpty()) {
                const QString label =
                        QStringLiteral("%1 (%2)").arg(set.label).arg(
                                set.keypoints.size());
                QRect labelRect(static_cast<int>(set.x1),
                                static_cast<int>(set.y1) - font.pixelSize() - 6,
                                std::max(20, label.size() * font.pixelSize()),
                                font.pixelSize() + 6);
                labelRect.moveLeft(std::clamp(
                        labelRect.left(), 2,
                        std::max(2, out.width() - labelRect.width() - 2)));
                if (labelRect.top() < 2) {
                    labelRect.moveTop(static_cast<int>(set.y1) + 2);
                }
                p.fillRect(
                        labelRect.adjusted(0, 0, 4, 2).intersected(out.rect()),
                        color);
                p.setPen(Qt::white);
                p.drawText(labelRect.adjusted(2, 3, -2, -2), label);
            }
        }
        // Keypoints: white-ringed dot + prompt/score label.
        const qreal radius = std::max(3.0, out.height() / 250.0);
        for (const GKDKeypoint& kp : set.keypoints) {
            if (kp.score < minScore) continue;
            p.setPen(Qt::NoPen);
            p.setBrush(Qt::white);
            p.drawEllipse(QPointF(kp.x, kp.y), radius, radius);
            p.setBrush(color);
            p.drawEllipse(QPointF(kp.x, kp.y), radius * 0.62, radius * 0.62);

            const QString text =
                    QStringLiteral("%1 %2")
                            .arg(kp.prompt.isEmpty() ? QStringLiteral("kpt")
                                                     : kp.prompt)
                            .arg(kp.score, 0, 'f', 2);
            QRect textRect(
                    static_cast<int>(kp.x) + static_cast<int>(radius) + 2,
                    static_cast<int>(kp.y) - font.pixelSize() / 2,
                    std::max(20, text.size() * font.pixelSize()),
                    font.pixelSize() + 4);
            textRect.moveLeft(std::clamp(
                    textRect.left(), 2,
                    std::max(2, out.width() - textRect.width() - 2)));
            p.fillRect(textRect.adjusted(0, 0, 3, 1).intersected(out.rect()),
                       QColor(0, 0, 0, 160));
            p.setPen(Qt::white);
            p.drawText(textRect.adjusted(2, 1, -2, -1), text);
        }
    }
    p.end();
    return out;
}

}  // namespace GKDHelpers
