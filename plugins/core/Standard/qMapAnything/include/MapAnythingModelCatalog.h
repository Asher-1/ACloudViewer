// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once

#include <QString>
#include <QStringList>
#include <QVector>

/** Catalog entry mirroring aicore_map_anything_model_entry. */
struct MapAnythingModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    /** Model-variant grouping key shared by the 4 quantization rows of one
     *  checkpoint family ("vggt-omega-1b-512", ...). The dialog renders one
     *  page per distinct variant in catalog order. */
    QString variant;
    qint64 sizeBytes = 0;
    // Official-parity inference defaults from the AICore catalog (sources
    // documented on aicore_map_anything_model_entry).
    QString defaultResizeMode;  // balanced | max_size | square
    QString confFilter;         // normalized | percentile
    float defaultMinConf = 0.67f;
};

namespace MapAnythingHelpers {

/** Enumerate the published catalog from AICore (32 rows: 8 variants x 4
 *  quantizations, first variant = vggt-omega-512, the accuracy pick). */
QVector<MapAnythingModelEntry> catalogModels();

/** Distinct variant keys in catalog order (8 entries; index 0 is the dialog's
 *  first tab — the vggt-omega best-accuracy checkpoint family). */
QStringList catalogVariants();

/** Rows of one variant, in catalog order (4 quantizations). */
QVector<MapAnythingModelEntry> modelsForVariant(const QString& variant);

/** Index (within catalogModels()) of the catalog-declared default row
 *  (aicore_map_anything_model_default_index); -1 when unavailable. */
int defaultModelIndex();

/** Combo/list display label for one catalog row: displayName plus the
 *  quantization note (same em-dash contract as qYOLO's
 *  YOLOHelpers::modelDisplayLabel). The quant note is the only row-level
 *  discriminating fact — displayName is constant across the quantization
 *  rows of one checkpoint family. */
QString modelDisplayLabel(const MapAnythingModelEntry& entry);

/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, MapAnythingModelEntry* out);

/** Model cache directory for qMapAnything
 *  (aicore_map_anything_model_cache_dir = …/extract/map_anything_models). */
QString modelCacheDir();

}  // namespace MapAnythingHelpers
