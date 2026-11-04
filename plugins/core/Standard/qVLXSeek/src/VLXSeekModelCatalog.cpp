// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VLXSeekModelCatalog.h"

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QObject>

#ifdef AICore_ENABLED
#include <aicore/ocr_capi.h>
#include <aicore/vlx_seek_capi.h>
#endif

namespace VLXSeekHelpers {

QVector<VLXSeekModelEntry> catalogModels() {
    QVector<VLXSeekModelEntry> out;
#ifdef AICore_ENABLED
    const int count = aicore_vlx_seek_model_count();
    out.reserve(count);
    for (int i = 0; i < count; ++i) {
        const aicore_vlx_seek_model_entry* e = aicore_vlx_seek_model_at(i);
        if (!e) continue;
        VLXSeekModelEntry row;
        row.filename = QString::fromUtf8(e->filename);
        row.downloadUrl = QString::fromUtf8(e->download_url);
        row.displayName = QString::fromUtf8(e->display_name);
        row.family = QString::fromUtf8(e->family);
        row.role = e->role;
        row.quantization = QString::fromUtf8(e->quantization);
        row.sha256 = QString::fromUtf8(e->sha256);
        row.sizeBytes = e->size_bytes;
        out.push_back(row);
    }
#endif
    return out;
}

int catalogDefaultIndex() {
#ifdef AICore_ENABLED
    return aicore_vlx_seek_model_default_index();
#else
    return -1;
#endif
}

bool findModelByFilename(const QString& filename, VLXSeekModelEntry* out) {
    if (filename.isEmpty()) return false;
    for (const VLXSeekModelEntry& row : catalogModels()) {
        if (row.filename == filename) {
            if (out) *out = row;
            return true;
        }
    }
    return false;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_vlx_seek_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_vlx_seek_free_buffer(dir);
        return out;
    }
#endif
    return QString();
}

QStringList roleDisplayLabels(int role) {
    QStringList labels;
    for (const VLXSeekModelEntry& row : catalogModels()) {
        if (row.role != role) continue;
        labels.push_back(row.displayName);
    }
    return labels;
}

QJsonArray catalogJson() {
    QJsonArray rows;
    for (const VLXSeekModelEntry& row : catalogModels()) {
        rows.append(QJsonObject{
                {"filename", row.filename},
                {"display_name", row.displayName},
                {"family", row.family},
                {"role", row.role},
                {"quantization", row.quantization},
                {"sha256", row.sha256},
                {"size_bytes", row.sizeBytes},
        });
    }
    return rows;
}

QVector<int32_t> parseTargetIndexes(const QString& text, QString* error) {
    QVector<int32_t> out;
    // Plain QChar split with the default KeepEmptyParts: identical on
    // Qt5/Qt6 (the SplitBehavior enum diverges across versions).
    const QStringList parts = text.split(QLatin1Char(','));
    for (const QString& part : parts) {
        const QString trimmed = part.trimmed();
        if (trimmed.isEmpty()) continue;
        bool ok = false;
        const int value = trimmed.toInt(&ok);
        if (!ok || value < 0) {
            if (error) {
                *error = QObject::tr("Invalid target region index: %1")
                                 .arg(trimmed);
            }
            return {};
        }
        out.push_back(static_cast<int32_t>(value));
    }
    return out;
}

QVector<float> parseExternalBoxes(const QString& path, QString* error) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        if (error)
            *error = QObject::tr("Cannot open external boxes file: %1")
                             .arg(path);
        return {};
    }
    const QJsonDocument doc = QJsonDocument::fromJson(file.readAll());
    const QJsonArray rows = doc.array();
    QVector<float> boxes;
    boxes.reserve(rows.size() * 4);
    for (const QJsonValue& row : rows) {
        const QJsonArray coords = row.toArray();
        if (coords.size() != 4) {
            if (error)
                *error = QObject::tr(
                                 "External box must hold exactly "
                                 "[x1,y1,x2,y2]: %1")
                                 .arg(path);
            return {};
        }
        for (const QJsonValue& c : coords) {
            if (!c.isDouble()) {
                if (error)
                    *error = QObject::tr(
                                     "External box coordinates must be "
                                     "numbers: %1")
                                     .arg(path);
                return {};
            }
            boxes.push_back(static_cast<float>(c.toDouble()));
        }
    }
    // The C API cap mirrors upstream (--bbox-list, <= 100 proposals).
    constexpr int kMaxBoxes = 100;
    if (boxes.size() / 4 > kMaxBoxes) {
        if (error)
            *error = QObject::tr("External box list exceeds %1 entries: %2")
                             .arg(kMaxBoxes)
                             .arg(path);
        return {};
    }
    return boxes;
}

QString ocrModelDownloadUrl(const QString& filename) {
#ifdef AICore_ENABLED
    if (filename.isEmpty()) return QString();
    const QByteArray utf8 = filename.toUtf8();
    const aicore_ocr_model_entry* entry =
            aicore_ocr_model_by_filename(utf8.constData());
    if (entry && entry->download_url)
        return QString::fromUtf8(entry->download_url);
#endif
    return QString();
}

OcrProposerModels ocrProposerModels() {
    OcrProposerModels out;
#ifdef AICore_ENABLED
    // The OCR preset owns the det/rec pairing; "ppocr-v4-ch" is the default
    // PP-OCRv4 Chinese recipe. q8_0 is the catalog's best size trade and a
    // manifest-validated quantization row. Everything resolves through the
    // public OCR catalog API — the plugin keeps no model table of its own.
    const aicore_ocr_preset_entry* preset = nullptr;
    for (int i = 0; i < aicore_ocr_preset_count(); ++i) {
        const aicore_ocr_preset_entry* p = aicore_ocr_preset_at(i);
        if (p && p->id &&
            QLatin1String(p->id) == QLatin1String("ppocr-v4-ch")) {
            preset = p;
            break;
        }
    }
    if (!preset) return out;
    out.detFileName = QStringLiteral("%1-%2.gguf")
                              .arg(QString::fromUtf8(preset->det_stem),
                                   QStringLiteral("q8_0"));
    out.recFileName = QStringLiteral("%1-%2.gguf")
                              .arg(QString::fromUtf8(preset->rec_stem),
                                   QStringLiteral("q8_0"));
    char* dir = aicore_ocr_model_cache_dir();
    if (dir) {
        out.cacheDir = QString::fromUtf8(dir);
        aicore_ocr_free_buffer(dir);
    }
#endif
    return out;
}

bool ocrQuadsToBoxes(const float* quads, int quadCount, QVector<float>* boxes) {
    if (!quads || quadCount <= 0 || !boxes) return false;
    boxes->clear();
    // The VLX-Seek C API accepts at most 100 proposals (the built-in
    // WeDetect-Uni proposer supplies exactly 100); det output beyond the cap
    // is dropped instead of failing the run.
    constexpr int kMaxProposals = 100;
    for (int i = 0; i < quadCount && boxes->size() / 4 < kMaxProposals; ++i) {
        const float* quad = quads + size_t(i) * 8;
        float x1 = quad[0], y1 = quad[1], x2 = quad[0], y2 = quad[1];
        for (int corner = 1; corner < 4; ++corner) {
            x1 = qMin(x1, quad[corner * 2 + 0]);
            y1 = qMin(y1, quad[corner * 2 + 1]);
            x2 = qMax(x2, quad[corner * 2 + 0]);
            y2 = qMax(y2, quad[corner * 2 + 1]);
        }
        if (x2 - x1 <= 0.0f || y2 - y1 <= 0.0f) continue;  // degenerate
        *boxes << x1 << y1 << x2 << y2;
    }
    return !boxes->isEmpty();
}

}  // namespace VLXSeekHelpers
