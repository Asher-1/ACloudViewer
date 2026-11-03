// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "VLXSeekModelCatalog.h"

#include <QJsonArray>
#include <QJsonObject>
#include <QObject>

#ifdef AICore_ENABLED
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

}  // namespace VLXSeekHelpers
