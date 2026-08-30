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

/** A GGUF published on the qTrellis Hugging Face mirror
 *  (https://huggingface.co/Asher-1/Trellis2-models). The mirror carries the
 *  f16 flow variants that exceed the 2 GB GitHub release limit. */
struct HfModelInfo {
    QString filename;
    qint64 sizeBytes = 0;  // published LFS size: display + no-ledger
                           // fallback guard for presence checks
    QString sha256;        // HF LFS content fingerprint (hex, 64 chars)
};

namespace TrellisHelpers {

/** Enumerate the published catalog from AICore. */
QVector<TrellisModelEntry> catalogModels();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, TrellisModelEntry* out);
/** Return all entries with the given role ("dino", "ss_flow", ...). */
QVector<TrellisModelEntry> modelsByRole(const QString& role);

/** Look up a file on the HF mirror (all f16/q8 variants). Returns false
 *  for files not published there. */
bool hfModelInfo(const QString& filename, HfModelInfo* out);
/** Direct download URL on the HF mirror (empty when not published). */
QString hfDownloadUrl(const QString& filename);
/** True when path holds a valid GGUF whose size matches the published
 *  mirror size for filename (falls back to the generic GGUF check when the
 *  file is not on the mirror). Lightweight (magic + size only, no full
 *  read) — this is the per-dialog presence check. */
bool isValidModelFile(const QString& path, const QString& filename);
/** Content-level verification against the published SHA-256. Reads the
 *  whole file — use only for one-shot checks (e.g. after a manual
 *  deployment), not for the per-dialog presence check. */
bool verifyModelFileSha256(const QString& path, const QString& filename);

/** The three built-in pipeline presets (Coarse / 512 / 1024). */
QVector<TrellisPreset> presets();

/** Resolve the preset's file list against the catalog into absolute paths.
 *  quantization selects the weight-precision chain: "q8" (default — every
 *  model that publishes a q8 variant uses it; the precision-sensitive
 *  decoders shape_dec / shape_enc / tex_dec have no q8 variant and always
 *  stay f16), "f16" (the full-half-precision reference chain) or "f32"
 *  (upstream's exact mode: the chaotic chain upgrades to full-f32 weights;
 *  those GGUFs are local conversions, not published on the mirror). Missing
 *  files are skipped; the caller decides whether that is fatal. */
QStringList resolvePresetFiles(const TrellisPreset& preset,
                               const QString& cacheDir,
                               const QString& quantization /* "q8"|"f16" */);

/** True when `filename` is one of the precision-sensitive decoders that
 *  always stay f16 regardless of the selected chain (sparse subdivision /
 *  UV decoding are not robust to Q8 weight rounding). */
bool isPrecisionSensitiveDecoder(const QString& filename);

/** Model cache directory for qTrellis (aicore_trellis_model_cache_dir). */
QString modelCacheDir();

/** Build a display label for a catalog entry. */
QString modelDisplayLabel(const TrellisModelEntry& entry);

}  // namespace TrellisHelpers
