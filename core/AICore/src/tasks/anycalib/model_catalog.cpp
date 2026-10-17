// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AnyCalib model catalog: published release assets for the fixed square
// 322x322 export of the AnyCalib v1.0.0 feed-forward network. The release
// assets are produced by core/AICore/tools/convert_anycalib_onnx_gguf.py
// from the pinned upstream ONNX (SHA-256 recorded in
// libs/Reconstruction/src/calibration/resources.h and re-verified inside the
// converter); their content digests are pinned in aicore/asset_digests.h.

#include <cstring>
#include <string>
#include <vector>

#include "aicore/anycalib_capi.h"
#include "common/capi_utils.hpp"
#include "common/model_cache.hpp"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "anycalib/";

// (release stem, user-facing role label, role, resolution). Stems match the
// converter output naming.
struct VariantInfo {
    const char* stem;
    const char* label;
    const char* role;
    const char* resolution;
};

static constexpr const VariantInfo kVariants[] = {
        {"anycalib_gen_v1.0.0_322x322", "Gen", "gen", "square322"},
};

static constexpr int kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

// Quantization suffixes — these must match the assets actually published in
// the anycalib release (verified by test_catalog_remote_assets). The Q8_0
// tier quantizes only the ViT transformer matmul weights (pre-transposed,
// flagged by anycalib.quant.matmul_pretransposed) and is gated by the
// cross-backend parity rows in validation_manifest.json.
static constexpr const char* kQuantSuffixes[] = {"f32", "f16", "q8_0"};

static constexpr int kQuantCount =
        sizeof(kQuantSuffixes) / sizeof(kQuantSuffixes[0]);

static constexpr const char* kQuantNotes[] = {
        "F32 \xe2\x80\x94 full precision reference",
        "F16 \xe2\x80\x94 half precision (recommended)",
        "Q8_0 \xe2\x80\x94 8-bit matmul weights (lightest)"};

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
    const char* role;
    const char* resolution;
};

static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    rows.reserve(static_cast<size_t>(kVariantCount * kQuantCount));
    for (int v = 0; v < kVariantCount; ++v) {
        for (int q = 0; q < kQuantCount; ++q) {
            const std::string filename = std::string(kVariants[v].stem) + "." +
                                         kQuantSuffixes[q] + ".gguf";
            const std::string url = std::string(kDownloadBase) + filename;
            const std::string display = std::string("AnyCalib ") +
                                        kVariants[v].label + " " +
                                        kQuantNotes[q];
            rows.push_back(
                    ModelRow{dupString(filename.c_str()),
                             dupString(url.c_str()), dupString(display.c_str()),
                             kVariants[v].role, kVariants[v].resolution});
        }
    }
    return rows;
}

const std::vector<ModelRow>& models() {
    static const std::vector<ModelRow> rows = buildModels();
    return rows;
}

}  // namespace

AICORE_CAPI int aicore_anycalib_model_count(void) {
    return static_cast<int>(models().size());
}

AICORE_CAPI const aicore_anycalib_model_entry* aicore_anycalib_model_at(
        int index) {
    if (index < 0 || index >= aicore_anycalib_model_count()) {
        return nullptr;
    }
    const ModelRow& row = models()[static_cast<size_t>(index)];
    // ModelRow and the public entry are layout-compatible by construction;
    // keep the reinterpret confined to this single point.
    return reinterpret_cast<const aicore_anycalib_model_entry*>(&row);
}

AICORE_CAPI const aicore_anycalib_model_entry*
aicore_anycalib_model_by_filename(const char* filename) {
    if (filename == nullptr) return nullptr;
    for (const ModelRow& row : models()) {
        if (std::strcmp(row.filename, filename) == 0) {
            return reinterpret_cast<const aicore_anycalib_model_entry*>(&row);
        }
    }
    return nullptr;
}

AICORE_CAPI char* aicore_anycalib_model_cache_dir(void) {
    return dupString(aicore::anycalib_model_cache_dir().c_str());
}
