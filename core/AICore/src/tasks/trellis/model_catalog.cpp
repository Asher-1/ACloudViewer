// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Published TRELLIS.2 GGUF catalog (cloudViewer_downloads trellis2-ggml
// release). The rmbg_* entries mirror the aicore_rmbg catalog (same release),
// so a TRELLIS pipeline can point at an already-cached RMBG model.

#include <cstring>
#include <string>
#include <vector>

#include "aicore/trellis_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "trellis2-ggml/";

// MSVC names the POSIX helper "_strdup"; keep a portable wrapper so the
// catalog builds warning-clean on all three platforms.
static char* dupString(const char* s) {
#ifdef _MSC_VER
    return _strdup(s);
#else
    return strdup(s);
#endif
}

struct ModelRow {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    const char* role;
};

// Build the model list at init time. Names/roles must match the assets
// actually published in the trellis2-ggml release.
static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    auto add = [&](const char* file, const char* display, const char* quant,
                   const char* license, const char* role) {
        std::string url = std::string(kDownloadBase) + file;
        rows.push_back({dupString(file), dupString(url.c_str()),
                        dupString(display), dupString(quant), dupString(license),
                        dupString(role)});
    };

    const char* kTrellisLicense = "MIT (TRELLIS.2-4B-GGUF / TRELLIS-image-large-GGUF)";
    const char* kDinoLicense =
            "DINOv3 License (built with DINOv3)";
    const char* kRmbgLicense =
            "CC BY-NC 4.0 (non-commercial); commercial license from BRIA";

    // Conditioning encoder.
    add("dino_f16.gguf", "DINOv3 ViT-L/16 f16", "F16 \xe2\x80\x94 half precision",
        kDinoLicense, "dino");
    add("dino_q8.gguf", "DINOv3 ViT-L/16 q8", "Q8 \xe2\x80\x94 8-bit quant (recommended)",
        kDinoLicense, "dino");

    // Sparse-structure stage (required for every quality).
    add("ss_flow_q8.gguf", "Sparse-structure flow q8", "Q8 \xe2\x80\x94 8-bit quant",
        kTrellisLicense, "ss_flow");
    add("ss_dec_f16.gguf", "Occupancy decoder f16", "F16 \xe2\x80\x94 half precision",
        kTrellisLicense, "ss_dec");
    add("ss_dec_q8.gguf", "Occupancy decoder q8", "Q8 \xe2\x80\x94 8-bit quant",
        kTrellisLicense, "ss_dec");

    // Shape stage: 512 fine + 1024 cascade.
    add("slat_flow_q8.gguf", "Shape-SLAT flow 512 q8", "Q8 \xe2\x80\x94 8-bit quant",
        kTrellisLicense, "slat_flow");
    add("slat_flow_1024_q8.gguf", "Shape-SLAT flow 1024 q8", "Q8 \xe2\x80\x94 8-bit quant",
        kTrellisLicense, "slat_flow_hr");
    add("shape_dec_f16.gguf", "Shape decoder f16", "F16 \xe2\x80\x94 half precision",
        kTrellisLicense, "shape_dec");

    // PBR texturing stage.
    add("shape_enc_f16.gguf", "Shape encoder f16", "F16 \xe2\x80\x94 half precision",
        kTrellisLicense, "shape_enc");
    add("tex_dec_f16.gguf", "Texture decoder f16", "F16 \xe2\x80\x94 half precision",
        kTrellisLicense, "tex_dec");
    add("tex_slat_flow_512_q8.gguf", "Texture flow 512 q8", "Q8 \xe2\x80\x94 8-bit quant",
        kTrellisLicense, "tex_flow");
    add("tex_slat_flow_1024_q8.gguf", "Texture flow 1024 q8", "Q8 \xe2\x80\x94 8-bit quant",
        kTrellisLicense, "tex_flow_hr");

    // AI background removal (shared with the aicore_rmbg catalog).
    add("rmbg_f32.gguf", "RMBG-2.0 f32", "F32 \xe2\x80\x94 full precision reference",
        kRmbgLicense, "rmbg");
    add("rmbg_f16.gguf", "RMBG-2.0 f16", "F16 \xe2\x80\x94 half precision (recommended)",
        kRmbgLicense, "rmbg");
    add("rmbg_q8.gguf", "RMBG-2.0 q8", "Q8 \xe2\x80\x94 8-bit quant, best accuracy/size trade",
        kRmbgLicense, "rmbg");

    return rows;
}

static const std::vector<ModelRow> kModels = buildModels();

static int modelCount() { return static_cast<int>(kModels.size()); }

static aicore_trellis_model_entry toEntry(const ModelRow& row) {
    return {row.filename, row.download_url, row.display_name, row.quant_note,
            row.license_note, row.role};
}

}  // namespace

AICORE_CAPI int aicore_trellis_model_count(void) { return modelCount(); }

AICORE_CAPI const aicore_trellis_model_entry* aicore_trellis_model_at(int index) {
    static thread_local aicore_trellis_model_entry entry{};
    if (index < 0 || index >= modelCount()) return nullptr;
    entry = toEntry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI const aicore_trellis_model_entry* aicore_trellis_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_trellis_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_trellis_model_download_base(void) {
    return kDownloadBase;
}
