// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TrellisModelCatalog.h"

#include <QDir>

#include "aicore/trellis_capi.h"
#include "ecvModelDownloader.h"

namespace {

// Hugging Face mirror — the single download source for all qTrellis GGUF
// models (both f16 and q8 variants). Every file is published on
// https://huggingface.co/Asher-1/Trellis2-models with its LFS metadata
// (size + SHA-256 oid). The GitHub trellis2-ggml release is kept for
// legacy fallback reference but is no longer queried by the plugin.
static const char* kHfRepoId = "Asher-1/Trellis2-models";

struct HfRow {
    const char* filename;
    qint64 sizeBytes;
    const char* sha256;
};

// Keep in sync with the published tree; a missing/renamed file simply makes
// hfModelInfo() return false and the caller falls back to the GitHub URL.
static const HfRow kHfModels[] = {
        {"dino_f16.gguf", 606992192LL,
         "385d8186a38a2328ec740fb2ac1f33f9194d8774efc7ccafd4aa2e51cf5f6450"},
        {"dino_q8.gguf", 323876672LL,
         "7da7e92438b0478a10a67d12a1e4439c9a54142d70c073fbe298dda1db10de53"},
        {"ss_flow_f16.gguf", 2615168864LL,
         "1dded5b74237d24e6876a642a26f90b43742e3554418573860f810e3bbe61e8c"},
        {"ss_flow_q8.gguf", 1418183264LL,
         "a75ab3b3c225bc62b7b33c54fee9d92e936c270ad5579daad4a6c8a3919a8d03"},
        {"ss_dec_f16.gguf", 147379616LL,
         "9c2210b7ed830fdc8286961a8189878ff5bcfd3bfc83ab4eacee005d293d2185"},
        {"ss_dec_q8.gguf", 147379616LL,
         "fe390843dcd2ca68fdb3d80bae0a2c9992d083b56844cfed3bcdfc2017a179ef"},
        {"slat_flow_f16.gguf", 2615319424LL,
         "2f94bad7b1c524ad8c01943bc38fcc0c314e7d482ce896f3c6e96eb6e7cec15c"},
        {"slat_flow_q8.gguf", 1418253184LL,
         "fedcc106efed4eb5469af4df8f380004271164a680ff642cd5bc6d614898c92a"},
        {"slat_flow_1024_f16.gguf", 2630208384LL,
         "e4cccf387fb31143eb000213e88b5c820f75cfea660e65914408a7329c118249"},
        {"slat_flow_1024_q8.gguf", 1418253184LL,
         "26577944aed86c270773262b13503a3a341b2d07bd38e19d879474c51a05c29d"},
        {"shape_dec_f16.gguf", 948745408LL,
         "6fe53f1d7763dabf7c8d72bc38f4053d87fde6f65bf17a9d378d27edb39d3530"},
        {"shape_enc_f16.gguf", 709034048LL,
         "3ec80ff580987fcdb9bc594fc8b6fda890d63101ca442eb2b26f5dc315e8696c"},
        {"tex_dec_f16.gguf", 948713856LL,
         "afd304f4dfcb8c94df851b85519b415b99f04070f7d29de1320c50631b1be4e0"},
        {"tex_slat_flow_512_f16.gguf", 2615421184LL,
         "89a081b7f5487a5b31f03d240e4d959a56db0cc2c46c327230097a2554da52ae"},
        {"tex_slat_flow_512_q8.gguf", 1418308864LL,
         "48f3f023ac24c76fd498ec7914dadbdd644b9d260248b293efcbbd905b75f191"},
        {"tex_slat_flow_1024_f16.gguf", 2615421184LL,
         "bbb55b0910c7929aac5e0612a9bb15113837a2c674cafb9f0f170eda8b5558a8"},
        {"tex_slat_flow_1024_q8.gguf", 1418308864LL,
         "24b2cab2429604aa7264e18baa2a86071e6017a74493ce5b5afd2e51c8a3cbf5"},
        {"rmbg_f32.gguf", 882846304LL,
         "73fa93582743128e392b6e5b6be821e5b67361dcd5a5c0deca0ae4077e4c0ddd"},
        {"rmbg_f16.gguf", 441451648LL,
         "50aaf0c7570df97b3767909394d9a63c93effe9f01dc863b17fa69d5f76eb8e3"},
        {"rmbg_q8.gguf", 258974848LL,
         "a2f432614d91057614c59745d40a1770b1a84455f5f853174629b05fc7c8e079"},
};

}  // namespace

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
    for (const HfRow& row : kHfModels) {
        if (filename == QLatin1String(row.filename)) {
            if (out) {
                out->filename = filename;
                out->sizeBytes = row.sizeBytes;
                out->sha256 = QString::fromLatin1(row.sha256);
            }
            return true;
        }
    }
    return false;
}

QString hfDownloadUrl(const QString& filename) {
    HfModelInfo info;
    if (!hfModelInfo(filename, &info)) return {};
    return ecvModelDownloader::hfDownloadUrl(QString::fromLatin1(kHfRepoId),
                                             filename)
            .toString();
}

bool isValidModelFile(const QString& path, const QString& filename) {
    HfModelInfo info;
    if (!hfModelInfo(filename, &info)) {
        // Not published on the mirror: fall back to the generic GGUF check
        // so previously supported (GitHub-only) files keep validating.
        return ecvModelDownloader::isValidCachedFile(path);
    }
    // Lightweight presence check: exact size + GGUF magic. A full SHA-256
    // pass over multi-GB files on every dialog refresh would be too slow;
    // content-level verification happens at download time (streamed) or via
    // verifyModelFileSha256() for manual deployments.
    return ecvModelDownloader::isValidCachedFile(path, 64 * 1024, true,
                                                 info.sizeBytes);
}

bool verifyModelFileSha256(const QString& path, const QString& filename) {
    HfModelInfo info;
    if (!hfModelInfo(filename, &info)) return false;
    return ecvModelDownloader::isValidCachedFile(
            path, 64 * 1024, true, info.sizeBytes, info.sha256.toLatin1());
}

QVector<TrellisPreset> presets() {
    QVector<TrellisPreset> out;
    // Default file lists use the f16 variants throughout (upstream's
    // recommended precision). The q8 alternatives remain selectable via the
    // dialog's variant combos (resolvePresetFiles substitutes the names).
    out.append({QStringLiteral("Coarse 64\u00b3 preview"),
                QStringLiteral("Fast occupancy preview (~3.4 GB, f16): dino + "
                               "ss_flow + ss_dec"),
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
    out.append(
            {QStringLiteral("Standard 512 + PBR (recommended)"),
             QStringLiteral(
                     "512\u00b3 fine dual-grid with PBR texturing (~11.2 GB, "
                     "f16)"),
             fine512});
    out.append(
            {QStringLiteral("Full 1024 cascade + PBR"),
             QStringLiteral("1024\u00b3 cascade with PBR texturing (~16.5 GB, "
                            "f16)"),
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

QStringList resolvePresetFiles(const TrellisPreset& preset,
                               const QString& cacheDir,
                               const QString& dinoVariant,
                               const QString& ssDecVariant) {
    QStringList out;
    for (const QString& file : preset.files) {
        // Variant selection: the preset lists the default f16 names, but the
        // dialog may prefer the smaller q8 dino / ss_dec.
        QString actual = file;
        if (file == QStringLiteral("dino_f16.gguf") &&
            dinoVariant == QStringLiteral("dino_q8")) {
            actual = QStringLiteral("dino_q8.gguf");
        } else if (file == QStringLiteral("ss_dec_f16.gguf") &&
                   ssDecVariant == QStringLiteral("ss_dec_q8")) {
            actual = QStringLiteral("ss_dec_q8.gguf");
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

QString modelDisplayLabel(const TrellisModelEntry& entry) {
    return entry.displayName;
}

}  // namespace TrellisHelpers
