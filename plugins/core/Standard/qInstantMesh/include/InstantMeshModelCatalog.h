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

/** Catalog entry mirroring aicore_instantmesh_model_entry. */
struct InstantMeshModelEntry {
    QString filename;
    QString downloadUrl;
    QString displayName;
    QString quantNote;
    QString licenseNote;
    QString role;  // "cond", "unet", "vae", "dino", "transformer",
                   // "synthesizer"
    qint64 sizeBytes = 0;
    QString sha256;
};

/** A named preset: the file set needed for one pipeline quality. */
struct InstantMeshPreset {
    QString name;
    QString description;
    QStringList files;  // GGUF filenames in aicore_instantmesh_model_paths
                        // field order (cond, unet, vae, dino, transformer,
                        // synthesizer); empty slot = omit that model
};

/** A GGUF published on the HF mirror (https://huggingface.co/Asher-1/
 *  InstantMesh-GUFF), read from the AICore catalog rather than maintained
 *  as a second table. */
struct HfInstantMeshInfo {
    QString filename;
    qint64 sizeBytes = 0;  // published LFS size
    QString sha256;        // HF LFS content fingerprint (hex, 64 chars)
};

namespace InstantMeshHelpers {

/** Enumerate the published catalog from AICore. */
QVector<InstantMeshModelEntry> catalogModels();
/** Lookup by GGUF filename; returns false when unknown. */
bool findModelByFilename(const QString& filename, InstantMeshModelEntry* out);
/** Return all entries with the given role. */
QVector<InstantMeshModelEntry> modelsByRole(const QString& role);

/** Look up a file on the HF mirror. Returns false for files not published
 *  in the AICore catalog. */
bool hfModelInfo(const QString& filename, HfInstantMeshInfo* out);
/** Direct download URL on the HF mirror (empty when not published). */
QString hfDownloadUrl(const QString& filename);
/** Base URL of the published HF mirror (ends with '/'). */
QString hfMirrorUrl();
/** True when path holds a valid GGUF whose size matches the published
 *  mirror size for filename (magic + size only, no full read). */
bool isValidModelFile(const QString& path, const QString& filename);
/** Content-level verification against the published SHA-256. */
bool verifyModelFileSha256(const QString& path, const QString& filename);

/** The built-in pipeline presets (Full / Geometry-only). */
QVector<InstantMeshPreset> presets();

/** Resolve the preset's file list against the catalog into absolute paths.
 *  quantization selects the weight-precision chain: "f16" (recommended —
 *  every model that publishes an f16 variant uses it; the Zero123++ cond
 *  constants only publish f32), "q8" (dino/transformer/synthesizer take q8;
 *  the Zero123++ trio has no q8 variant and stays f16/f32) or "f32"
 *  (the full exact-reference chain). Missing files are skipped; the caller
 *  decides whether that is fatal. */
QStringList resolvePresetFiles(const InstantMeshPreset& preset,
                               const QString& cacheDir,
                               const QString& quantization);

/** Model cache directory for qInstantMesh
 *  (aicore_instantmesh_model_cache_dir). */
QString modelCacheDir();
/** Shared RMBG dependency cache (aicore_rmbg_model_cache_dir). */
QString rmbgModelCacheDir();
/** Build a display label for a catalog entry. */
QString modelDisplayLabel(const InstantMeshModelEntry& entry);

}  // namespace InstantMeshHelpers
