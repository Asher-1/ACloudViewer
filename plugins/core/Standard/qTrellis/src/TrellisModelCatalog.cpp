// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "TrellisModelCatalog.h"

#include <QDir>

#include "aicore/trellis_capi.h"

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

QVector<TrellisPreset> presets() {
    QVector<TrellisPreset> out;
    out.append({QStringLiteral("Coarse 64\u00b3 preview"),
                QStringLiteral("Fast occupancy preview (~4.5 GB): dino + "
                               "ss_flow + ss_dec"),
                {QStringLiteral("dino_f16.gguf"),
                 QStringLiteral("ss_flow_q8.gguf"),
                 QStringLiteral("ss_dec_f16.gguf")}});
    // The file lists below MUST stay in aicore_trellis_model_paths field
    // order (dino, ss_flow, ss_dec, slat_flow, slat_hr_flow, shape_dec,
    // shape_enc, tex_dec, tex_flow, tex_flow_hr): TrellisWorker assigns the
    // resolved list by index. A preset that omits a field keeps an empty
    // string placeholder so later fields do not shift.
    QStringList fine512;
    fine512 << QStringLiteral("dino_f16.gguf")
            << QStringLiteral("ss_flow_q8.gguf")
            << QStringLiteral("ss_dec_f16.gguf")
            << QStringLiteral("slat_flow_q8.gguf")
            << QString()  // slat_hr_flow (1024) — not in this preset
            << QStringLiteral("shape_dec_f16.gguf")
            << QStringLiteral("shape_enc_f16.gguf")
            << QStringLiteral("tex_dec_f16.gguf")
            << QStringLiteral("tex_slat_flow_512_q8.gguf")
            << QString();  // tex_flow_hr (1024) — not in this preset
    out.append(
            {QStringLiteral("Standard 512 + PBR (recommended)"),
             QStringLiteral(
                     "512\u00b3 fine dual-grid with PBR texturing (~7.9 GB)"),
             fine512});
    out.append(
            {QStringLiteral("Full 1024 cascade + PBR"),
             QStringLiteral("1024\u00b3 cascade with PBR texturing (~9.3 GB)"),
             {QStringLiteral("dino_f16.gguf"),
              QStringLiteral("ss_flow_q8.gguf"),
              QStringLiteral("ss_dec_f16.gguf"),
              QStringLiteral("slat_flow_q8.gguf"),
              QStringLiteral("slat_flow_1024_q8.gguf"),
              QStringLiteral("shape_dec_f16.gguf"),
              QStringLiteral("shape_enc_f16.gguf"),
              QStringLiteral("tex_dec_f16.gguf"),
              QStringLiteral("tex_slat_flow_512_q8.gguf"),
              QStringLiteral("tex_slat_flow_1024_q8.gguf")}});
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
