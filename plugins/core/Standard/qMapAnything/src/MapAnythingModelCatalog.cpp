// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "MapAnythingModelCatalog.h"

#include <QDir>
#include <cstdlib>

#ifdef AICore_ENABLED
#include "aicore/map_anything_capi.h"
#endif

QVector<MapAnythingModelEntry> MapAnythingHelpers::catalogModels() {
    QVector<MapAnythingModelEntry> models;
#ifdef AICore_ENABLED
    const int count = aicore_map_anything_model_count();
    models.reserve(count);
    for (int i = 0; i < count; ++i) {
        const auto* entry = aicore_map_anything_model_at(i);
        if (entry == nullptr) continue;
        MapAnythingModelEntry m;
        m.filename = QString::fromUtf8(entry->filename);
        m.downloadUrl = QString::fromUtf8(entry->download_url);
        m.displayName = QString::fromUtf8(entry->display_name);
        m.quantNote = QString::fromUtf8(entry->quant_note);
        m.licenseNote = QString::fromUtf8(entry->license_note);
        m.variant = QString::fromUtf8(entry->variant);
        m.sizeBytes = entry->size_bytes;
        m.defaultResizeMode = QString::fromUtf8(entry->default_resize_mode);
        m.confFilter = QString::fromUtf8(entry->conf_filter);
        m.defaultMinConf = entry->default_min_conf;
        models.append(m);
    }
#endif
    return models;
}

QStringList MapAnythingHelpers::catalogVariants() {
    QStringList variants;
    for (const auto& model : catalogModels()) {
        if (!variants.contains(model.variant)) variants.append(model.variant);
    }
    return variants;
}

QVector<MapAnythingModelEntry> MapAnythingHelpers::modelsForVariant(
        const QString& variant) {
    QVector<MapAnythingModelEntry> models;
    for (const auto& model : catalogModels()) {
        if (model.variant == variant) models.append(model);
    }
    return models;
}

int MapAnythingHelpers::defaultModelIndex() {
#ifdef AICore_ENABLED
    return aicore_map_anything_model_default_index();
#else
    return -1;
#endif
}

QString MapAnythingHelpers::modelDisplayLabel(
        const MapAnythingModelEntry& entry) {
    QString label = entry.displayName;
    if (!entry.quantNote.isEmpty() && !label.contains(entry.quantNote)) {
        label += QStringLiteral(" \u2014 ") + entry.quantNote;
    }
    return label;
}

bool MapAnythingHelpers::findModelByFilename(const QString& filename,
                                             MapAnythingModelEntry* out) {
    for (const auto& model : catalogModels()) {
        if (model.filename == filename) {
            if (out != nullptr) *out = model;
            return true;
        }
    }
    return false;
}

QString MapAnythingHelpers::modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_map_anything_model_cache_dir();
    const QString path = QString::fromUtf8(dir != nullptr ? dir : "");
    std::free(dir);
    if (!path.isEmpty()) {
        QDir().mkpath(path);
        return path;
    }
#endif
    return QString();
}
