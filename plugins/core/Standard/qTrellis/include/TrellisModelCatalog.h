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
#include <cstdint>

/** Catalog entry mirroring aicore_trellis_model_entry. */
struct TrellisModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    QString role;  // "dino", "ss_flow", "ss_dec", "slat_flow", ...
};

/** A named preset: the file set needed for one pipeline quality. */
struct TrellisPreset {
    QString name;
    QString description;
    QStringList files;  // GGUF filenames (without rmbg_* — optional add-on)
};

namespace TrellisHelpers {

/** Enumerate the published catalog from AICore. */
QVector<TrellisModelEntry> catalogModels();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, TrellisModelEntry* out);
/** Return all entries with the given role ("dino", "ss_flow", ...). */
QVector<TrellisModelEntry> modelsByRole(const QString& role);

/** The three built-in pipeline presets (Coarse / 512 / 1024). */
QVector<TrellisPreset> presets();

/** Resolve the preset's file list against the catalog into absolute paths.
 *  Missing files are skipped; the caller decides whether that is fatal. */
QStringList resolvePresetFiles(const TrellisPreset& preset,
                               const QString& cacheDir,
                               const QString& dinoVariant /* "dino_q8" | "dino_f16" */,
                               const QString& ssDecVariant /* "ss_dec_q8" | "ss_dec_f16" */);

/** Model cache directory for qTrellis (aicore_trellis_model_cache_dir). */
QString modelCacheDir();

/** Build a display label for a catalog entry. */
QString modelDisplayLabel(const TrellisModelEntry& entry);

}  // namespace TrellisHelpers
