// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "InstantMeshModelCatalog.h"

#include <QCryptographicHash>
#include <QDir>

#include "aicore/instantmesh_capi.h"
#include "aicore/rmbg_capi.h"
#include "ecvAssetIntegrity.h"

namespace InstantMeshHelpers {

QVector<InstantMeshModelEntry> catalogModels() {
    QVector<InstantMeshModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_instantmesh_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_instantmesh_model_entry* e =
                aicore_instantmesh_model_at(i);
        if (!e || !e->filename) continue;
        InstantMeshModelEntry entry;
        entry.filename = QString::fromUtf8(e->filename);
        entry.downloadUrl = QString::fromUtf8(e->download_url);
        entry.displayName = QString::fromUtf8(e->display_name);
        entry.quantNote = QString::fromUtf8(e->quant_note);
        entry.licenseNote = QString::fromUtf8(e->license_note);
        entry.role = QString::fromUtf8(e->role ? e->role : "");
        entry.sizeBytes = static_cast<qint64>(e->size_bytes);
        entry.sha256 = QString::fromUtf8(e->sha256 ? e->sha256 : "");
        out.append(entry);
    }
#else
    (void)0;
#endif
    return out;
}

bool findModelByFilename(const QString& filename, InstantMeshModelEntry* out) {
    const QVector<InstantMeshModelEntry> all = catalogModels();
    for (const InstantMeshModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QVector<InstantMeshModelEntry> modelsByRole(const QString& role) {
    QVector<InstantMeshModelEntry> out;
    const QVector<InstantMeshModelEntry> all = catalogModels();
    for (const InstantMeshModelEntry& e : all) {
        if (e.role == role) out.append(e);
    }
    return out;
}

bool hfModelInfo(const QString& filename, HfInstantMeshInfo* out) {
#ifdef AICore_ENABLED
    const QByteArray name = filename.toUtf8();
    const aicore_instantmesh_model_entry* entry =
            aicore_instantmesh_model_by_filename(name.constData());
    if (!entry || !entry->sha256 || entry->size_bytes == 0) return false;
    if (out) {
        out->filename = filename;
        out->sizeBytes = static_cast<qint64>(entry->size_bytes);
        out->sha256 = QString::fromUtf8(entry->sha256);
    }
    return true;
#else
    Q_UNUSED(filename);
    Q_UNUSED(out);
    return false;
#endif
}

QString hfDownloadUrl(const QString& filename) {
#ifdef AICore_ENABLED
    const QByteArray name = filename.toUtf8();
    const aicore_instantmesh_model_entry* entry =
            aicore_instantmesh_model_by_filename(name.constData());
    return entry && entry->download_url ? QString::fromUtf8(entry->download_url)
                                        : QString();
#else
    Q_UNUSED(filename);
    return {};
#endif
}

QString hfMirrorUrl() {
    const QVector<InstantMeshModelEntry> all = catalogModels();
    for (const InstantMeshModelEntry& e : all) {
        const int slash = e.downloadUrl.lastIndexOf(QLatin1Char('/'));
        if (e.downloadUrl.startsWith(QStringLiteral("https://")) && slash > 0) {
            return e.downloadUrl.left(slash + 1);
        }
    }
    return QString();
}

bool isValidModelFile(const QString& path, const QString& filename) {
    HfInstantMeshInfo info;
    if (!hfModelInfo(filename, &info)) return false;
    // Lightweight presence check: integrity-ledger stat-trust (the HF LFS
    // oid is the pinned digest), falling back to the exact published size.
    return ecvAssetIntegrity::isVerified(
            path, {QCryptographicHash::Sha256, info.sha256.toLatin1()},
            64 * 1024, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
            info.sizeBytes);
}

bool verifyModelFileSha256(const QString& path, const QString& filename) {
    HfInstantMeshInfo info;
    if (!hfModelInfo(filename, &info)) return false;
    return ecvAssetIntegrity::verifyNow(
            path, {QCryptographicHash::Sha256, info.sha256.toLatin1()});
}

QVector<InstantMeshPreset> presets() {
    QVector<InstantMeshPreset> out;
    // File lists are in aicore_instantmesh_model_paths field order (cond,
    // unet, vae, dino, transformer, synthesizer); resolvePresetFiles()
    // substitutes the selected quantization variants. The Zero123++ cond
    // constants only publish f32; the UNet/VAE publish f16/f32.
    out.append({QStringLiteral("Full multi-view (recommended)"),
                QStringLiteral("Zero123++ 6-view diffusion + FlexiCubes mesh "
                               "(~5.2 GB f16 / ~4.0 GB q8-mix / ~8.9 GB f32)"),
                {QStringLiteral("zero123pp_cond_f32.gguf"),
                 QStringLiteral("zero123pp_unet_f16.gguf"),
                 QStringLiteral("zero123pp_vae_f16.gguf"),
                 QStringLiteral("dino_f16.gguf"),
                 QStringLiteral("lrm_transformer_f16.gguf"),
                 QStringLiteral("synthesizer_f16.gguf")}});
    out.append(
            {QStringLiteral("Geometry only (replicated views)"),
             QStringLiteral("No multi-view diffusion: the input image is "
                            "replicated 6x as the view condition (upstream "
                            "benchmark path; ~1.2 GB q8)"),
             {QString(), QString(), QString(), QStringLiteral("dino_f16.gguf"),
              QStringLiteral("lrm_transformer_f16.gguf"),
              QStringLiteral("synthesizer_f16.gguf")}});
    return out;
}

QStringList resolvePresetFiles(const InstantMeshPreset& preset,
                               const QString& cacheDir,
                               const QString& quantization) {
    const bool wantQ8 = quantization == QStringLiteral("q8");
    const bool wantF32 = quantization == QStringLiteral("f32");
    QStringList out;
    for (const QString& file : preset.files) {
        QString actual = file;
        if (!file.isEmpty() && file.endsWith(QStringLiteral("_f16.gguf"))) {
            const QString stem =
                    file.left(file.size() - QStringLiteral("f16.gguf").size());
            if (wantQ8) {
                // Only the InstantMesh trio publishes q8; the Zero123++
                // UNet/VAE have no q8 variant and stay f16 (cond stays f32).
                const QString q8 = stem + QStringLiteral("q8.gguf");
                HfInstantMeshInfo info;
                if (hfModelInfo(q8, &info)) actual = q8;
            } else if (wantF32) {
                const QString f32 = stem + QStringLiteral("f32.gguf");
                HfInstantMeshInfo info;
                if (hfModelInfo(f32, &info)) actual = f32;
            }
        }
        if (actual.isEmpty()) {
            // Keep the placeholder slot: the caller maps the resolved list by
            // index onto aicore_instantmesh_model_paths, so an omitted field
            // must stay in position (empty string = "omit this model").
            out << QString();
            continue;
        }
        out << cacheDir + QDir::separator() + actual;
    }
    return out;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_instantmesh_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_instantmesh_free_buffer(dir);
        return out;
    }
#endif
    return QString();
}

QString rmbgModelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_rmbg_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_rmbg_free_buffer(dir);
        return out;
    }
#endif
    return QString();
}

QString modelDisplayLabel(const InstantMeshModelEntry& entry) {
    return entry.displayName;
}

}  // namespace InstantMeshHelpers
