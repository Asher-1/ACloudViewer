// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TrellisModelCatalog.h"

#include <QCryptographicHash>
#include <QDir>

#include "aicore/rmbg_capi.h"
#include "aicore/trellis_capi.h"
#include "ecvAssetIntegrity.h"

namespace TrellisHelpers {

QVector<TrellisModelEntry> catalogModels() {
    QVector<TrellisModelEntry> out;
#ifdef AICore_ENABLED
    const int n = aicore_trellis_model_count();
    out.reserve(n > 0 ? n : 0);
    for (int i = 0; i < n; ++i) {
        const aicore_trellis_model_entry* e = aicore_trellis_model_at(i);
        if (!e || !e->filename) continue;
        TrellisModelEntry entry;
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

bool findModelByFilename(const QString& filename, TrellisModelEntry* out) {
    const QVector<TrellisModelEntry> all = catalogModels();
    for (const TrellisModelEntry& e : all) {
        if (e.filename == filename) {
            if (out) *out = e;
            return true;
        }
    }
    return false;
}

QVector<TrellisModelEntry> modelsByRole(const QString& role) {
    QVector<TrellisModelEntry> out;
    const QVector<TrellisModelEntry> all = catalogModels();
    for (const TrellisModelEntry& e : all) {
        if (e.role == role) out.append(e);
    }
    return out;
}

bool hfModelInfo(const QString& filename, HfModelInfo* out) {
#ifdef AICore_ENABLED
    const QByteArray name = filename.toUtf8();
    const aicore_trellis_model_entry* entry =
            aicore_trellis_model_by_filename(name.constData());
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
    const aicore_trellis_model_entry* entry =
            aicore_trellis_model_by_filename(name.constData());
    return entry && entry->download_url ? QString::fromUtf8(entry->download_url)
                                        : QString();
#else
    Q_UNUSED(filename);
    return {};
#endif
}

QString hfMirrorUrl() {
    // Derived from the catalog's own download URLs so the manual-recovery
    // hint cannot drift from the published mirror.
    const QVector<TrellisModelEntry> all = catalogModels();
    for (const TrellisModelEntry& e : all) {
        const int slash = e.downloadUrl.lastIndexOf(QLatin1Char('/'));
        if (e.downloadUrl.startsWith(QStringLiteral("https://")) && slash > 0) {
            return e.downloadUrl.left(slash + 1);
        }
    }
    return QString();
}

bool isValidModelFile(const QString& path, const QString& filename) {
    HfModelInfo info;
    if (!hfModelInfo(filename, &info)) return false;
    // Lightweight presence check: integrity-ledger stat-trust (the HF LFS
    // oid is the pinned digest), falling back to the exact published size
    // for files downloaded before the ledger existed. A full SHA-256 pass
    // over multi-GB files on every dialog refresh would be too slow;
    // content-level verification happens at download time (streamed) or via
    // verifyModelFileSha256() for manual deployments.
    return ecvAssetIntegrity::isVerified(
            path,
            {QCryptographicHash::Sha256,
             ecvAssetIntegrity::PinnedDigest(filename)},
            64 * 1024, true, ecvAssetIntegrity::OnMiss::CheapChecksOnly,
            info.sizeBytes);
}

bool verifyModelFileSha256(const QString& path, const QString& filename) {
    HfModelInfo info;
    if (!hfModelInfo(filename, &info)) return false;
    return ecvAssetIntegrity::verifyNow(
            path, {QCryptographicHash::Sha256, info.sha256.toLatin1()});
}

QVector<TrellisPreset> presets() {
    QVector<TrellisPreset> out;
    // The canonical file lists name the f16 GGUFs; resolvePresetFiles()
    // substitutes the q8 variants per the selected quantization chain
    // (default q8 — every model that publishes a q8 variant). The
    // precision-sensitive decoders (shape_dec / shape_enc / tex_dec) have
    // no q8 variant and always stay f16.
    out.append({QStringLiteral("Coarse 64\u00b3 preview"),
                QStringLiteral("Fast occupancy preview (~3.4 GB q8 / 3.4 GB "
                               "f16): dino + ss_flow + ss_dec"),
                {QStringLiteral("dino_f16.gguf"),
                 QStringLiteral("ss_flow_f16.gguf"),
                 QStringLiteral("ss_dec_f16.gguf")}});
    // The file lists below MUST stay in aicore_trellis_model_paths field
    // order (dino, ss_flow, ss_dec, slat_flow, slat_hr_flow, shape_dec,
    // shape_enc, tex_dec, tex_flow, tex_flow_hr): TrellisWorker assigns the
    // resolved list by index. A preset that omits a field keeps an empty
    // string placeholder so later fields do not shift.
    QStringList fine512;
    fine512 << QStringLiteral("dino_f16.gguf")
            << QStringLiteral("ss_flow_f16.gguf")
            << QStringLiteral("ss_dec_f16.gguf")
            << QStringLiteral("slat_flow_f16.gguf")
            << QString()  // slat_hr_flow (1024) — not in this preset
            << QStringLiteral("shape_dec_f16.gguf")
            << QStringLiteral("shape_enc_f16.gguf")
            << QStringLiteral("tex_dec_f16.gguf")
            << QStringLiteral("tex_slat_flow_512_f16.gguf")
            << QString();  // tex_flow_hr (1024) — not in this preset
    out.append({QStringLiteral("Standard 512 + PBR (recommended)"),
                QStringLiteral(
                        "512\u00b3 fine dual-grid with PBR texturing (~7.5 GB "
                        "q8 / ~11.2 GB f16)"),
                fine512});
    out.append(
            {QStringLiteral("Full 1024 cascade + PBR"),
             QStringLiteral("1024\u00b3 cascade with PBR texturing (~10.5 GB "
                            "q8 / ~16.5 GB f16)"),
             {QStringLiteral("dino_f16.gguf"),
              QStringLiteral("ss_flow_f16.gguf"),
              QStringLiteral("ss_dec_f16.gguf"),
              QStringLiteral("slat_flow_f16.gguf"),
              QStringLiteral("slat_flow_1024_f16.gguf"),
              QStringLiteral("shape_dec_f16.gguf"),
              QStringLiteral("shape_enc_f16.gguf"),
              QStringLiteral("tex_dec_f16.gguf"),
              QStringLiteral("tex_slat_flow_512_f16.gguf"),
              QStringLiteral("tex_slat_flow_1024_f16.gguf")}});
    return out;
}

bool isPrecisionSensitiveDecoder(const QString& filename) {
    // Sparse subdivision / UV decoding are not robust to Q8 weight rounding;
    // these roles never take the q8 chain (they also have no q8 variant
    // published). shape_dec is in a separate class: it stays f16 on q8 but
    // upgrades to f32 on the exact-mode f32 chain (isChaoticChainModel).
    return filename.startsWith(QStringLiteral("shape_dec_")) ||
           filename.startsWith(QStringLiteral("shape_enc_")) ||
           filename.startsWith(QStringLiteral("tex_dec_"));
}

bool isChaoticChainModel(const QString& filename) {
    // Models feeding the chaotic CFG samplers: the upstream f32 (exact)
    // mode upgrades exactly this set to full-f32 weights (the texture chain
    // stays f16 there — its noise affects appearance, not the voxel set).
    return filename.startsWith(QStringLiteral("dino_")) ||
           filename.startsWith(QStringLiteral("ss_flow_")) ||
           filename.startsWith(QStringLiteral("ss_dec_")) ||
           filename.startsWith(QStringLiteral("slat_flow_")) ||
           filename.startsWith(QStringLiteral("shape_dec_"));
}

QStringList resolvePresetFiles(const TrellisPreset& preset,
                               const QString& cacheDir,
                               const QString& quantization) {
    const bool wantQ8 = quantization == QStringLiteral("q8");
    const bool wantF32 = quantization == QStringLiteral("f32");
    QStringList out;
    for (const QString& file : preset.files) {
        QString actual = file;
        if (file.endsWith(QStringLiteral("_f16.gguf"))) {
            const QString stem =
                    file.left(file.size() - QStringLiteral("f16.gguf").size());
            if (wantQ8 && file.startsWith(QStringLiteral("ss_dec_"))) {
                // q8 chain (upstream 2026-08-30 CUDA+Vulkan e2e experiment):
                // ONLY ss_dec takes the q8 substitution. Everything feeding
                // the chaotic CFG samplers (dino cond, ss_flow, slat_flow)
                // plus the three VAE stages stays f16 — even a Q8_0 cond
                // error of ~7e-3 rel-L2 is amplified into completely
                // different voxel sets. ss_dec's q8 GGUF is itself a byte
                // clone of f16 (its dense conv3d layout [3,3,3,N] is not
                // Q8-blockable), so q8 currently saves nothing on this
                // pipeline and stays selectable for experiments only.
                const QString q8 = stem + QStringLiteral("q8.gguf");
                HfModelInfo info;
                if (hfModelInfo(q8, &info)) actual = q8;
            } else if (wantF32 && isChaoticChainModel(file)) {
                // f32 exact chain: upgrade the chaotic chain to the published
                // full-f32 weights. Texture-only files have no f32 release
                // and deliberately remain on canonical f16 weights.
                actual = stem + QStringLiteral("f32.gguf");
            }
        }
        if (actual.isEmpty()) {
            // Keep the placeholder slot: the caller maps the resolved list by
            // index onto aicore_trellis_model_paths, so an omitted field must
            // stay in position (empty string = "omit this model").
            out << QString();
            continue;
        }
        out << cacheDir + QDir::separator() + actual;
    }
    return out;
}

QString modelCacheDir() {
#ifdef AICore_ENABLED
    char* dir = aicore_trellis_model_cache_dir();
    if (dir) {
        const QString out = QString::fromUtf8(dir);
        aicore_trellis_free_buffer(dir);
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

QString modelCacheDirFor(const QString& filename) {
    TrellisModelEntry entry;
    return findModelByFilename(filename, &entry) &&
                           entry.role == QStringLiteral("rmbg")
                   ? rmbgModelCacheDir()
                   : modelCacheDir();
}

QString modelDisplayLabel(const TrellisModelEntry& entry) {
    return entry.displayName;
}

}  // namespace TrellisHelpers
