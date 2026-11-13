// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
#include "QuerySplatModelCatalog.h"

#include <aicore/querysplat_capi.h>

#include <QFileInfo>

namespace {

QuerySplatModelEntry toEntry(const aicore_querysplat_model_entry* e) {
    QuerySplatModelEntry out;
    if (!e) return out;
    out.filename = QString::fromUtf8(e->filename);
    out.downloadUrl = QString::fromUtf8(e->download_url);
    out.displayName = QString::fromUtf8(e->display_name);
    out.role = QString::fromUtf8(e->role);
    out.quantNote = QString::fromUtf8(e->quant_note);
    out.licenseNote = QString::fromUtf8(e->license_note);
    out.variant = QString::fromUtf8(e->variant);
    out.sizeBytes = e->size_bytes;
    return out;
}

}  // namespace

namespace QuerySplatHelpers {

QVector<QuerySplatModelEntry> catalogModels() {
    QVector<QuerySplatModelEntry> out;
    const int32_t n = aicore_querysplat_model_count();
    out.reserve(n);
    for (int32_t i = 0; i < n; ++i) {
        out.push_back(toEntry(aicore_querysplat_model_at(i)));
    }
    return out;
}

QVector<QuerySplatModelEntry> decoderModels() {
    QVector<QuerySplatModelEntry> out;
    for (const auto& m : catalogModels()) {
        if (m.role == QStringLiteral("decoder")) out.push_back(m);
    }
    return out;
}

int defaultDecoderIndex() { return 0; }

QString decoderCacheDir() {
    // resolved through the AICore catalog path helper (CLOUDVIEWER_DATA_ROOT
    // aware); fall back to the default decoder row's directory
    if (char* p = aicore_querysplat_model_resolve_cache_path(
                "querysplat-paper-q8_0.gguf")) {
        QString dir = QFileInfo(QString::fromUtf8(p)).absolutePath();
        aicore_querysplat_free_buffer(p);
        return dir;
    }
    return QString();
}

QString encoderDefaultPath() {
    const char* def = aicore_querysplat_model_default("encoder");
    return def ? QString::fromUtf8(def) : QString();
}

QString findModelByFilename(const QString& filename,
                            QuerySplatModelEntry* out) {
    for (const auto& m : catalogModels()) {
        if (m.filename == filename) {
            if (out) *out = m;
            return m.filename;
        }
    }
    return QString();
}

}  // namespace QuerySplatHelpers
