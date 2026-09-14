// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Published GKDT-L GGUF catalog. Source of truth: the Hugging Face repo
// Asher-1/GKD_GGUF (exact LFS sizes + SHA-256 pinned in
// aicore/asset_digests.h). The regression runner discovers models only through
// this catalog — never through local file listings.

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/gkd_capi.h"

namespace {

// Hugging Face resolve-main base (ecvModelDownloader follows the 302 to the
// CDN automatically; public repo, no token needed).
static constexpr const char* kDownloadBase =
        "https://huggingface.co/Asher-1/GKD_GGUF/resolve/main/";

struct ModelRow {
    const char* filename;
    const char* display_name;
    const char* quant_note;
    int64_t size_bytes;
};

// Order follows the published repo (gkd_fullset-q4_0.gguf is deprecated
// upstream and intentionally NOT cataloged); the default index points at the
// row whose quant_note carries the visible "(recommended)" marker (q4_K — the
// upstream recommendation: same 483 MiB footprint as the legacy q4_0 with
// clearly better accuracy and the fastest CPU config). The marker lives in
// the quant note, matching every other task's catalog convention.
static constexpr ModelRow kModels[] = {
        {"gkd_fullset-f16.gguf", "GKDT-L (F16)",
         "F16 \xe2\x80\x94 near-lossless, best Vulkan latency", 1779457504},
        {"gkd_fullset-f32.gguf", "GKDT-L (F32)",
         "F32 \xe2\x80\x94 full precision reference", 3550633440},
        {"gkd_fullset-q8_0.gguf", "GKDT-L (Q8_0)",
         "Q8_0 \xe2\x80\x94 near-lossless (<= 0.002 score diff)", 949218784},
        {"gkd_fullset-q4_K.gguf", "GKDT-L (Q4_K)",
         "Q4_K \xe2\x80\x94 483 MiB, exact coordinates (recommended)",
         506424800},
};

static constexpr int kModelCount = sizeof(kModels) / sizeof(kModels[0]);

static int default_index() {
    for (int i = 0; i < kModelCount; ++i) {
        if (std::strstr(kModels[i].quant_note, "(recommended)") != nullptr) {
            return i;
        }
    }
    return 0;
}

// Stable download-URL storage (the entry structs expose const char* that
// must stay valid for the process lifetime).
const std::vector<std::string>& catalog_urls() {
    static const std::vector<std::string> urls = [] {
        std::vector<std::string> v;
        v.reserve(kModelCount);
        for (const auto& m : kModels) {
            v.emplace_back(std::string(kDownloadBase) + m.filename);
        }
        return v;
    }();
    return urls;
}

const aicore_gkd_model_entry* catalog_rows() {
    static const std::vector<aicore_gkd_model_entry> rows = [] {
        std::vector<aicore_gkd_model_entry> v;
        v.reserve(kModelCount);
        const auto& urls = catalog_urls();
        for (int i = 0; i < kModelCount; ++i) {
            aicore_gkd_model_entry e{};
            e.filename = kModels[i].filename;
            e.download_url = urls[i].c_str();
            e.display_name = kModels[i].display_name;
            e.quant_note = kModels[i].quant_note;
            e.license_note =
                    "GKDT: academic research & educational use only "
                    "(ECCV 2026); commercial use prohibited";
            e.size_bytes = kModels[i].size_bytes;
            v.push_back(e);
        }
        return v;
    }();
    return rows.data();
}

}  // namespace

AICORE_CAPI int aicore_gkd_model_count(void) { return kModelCount; }

AICORE_CAPI const aicore_gkd_model_entry* aicore_gkd_model_at(int index) {
    if (index < 0 || index >= kModelCount) return nullptr;
    return catalog_rows() + index;
}

AICORE_CAPI int aicore_gkd_model_default_index(void) { return default_index(); }

AICORE_CAPI const aicore_gkd_model_entry* aicore_gkd_model_by_filename(
        const char* filename) {
    if (filename == nullptr) return nullptr;
    for (int i = 0; i < kModelCount; ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_gkd_model_at(i);
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_gkd_model_download_base(void) {
    return kDownloadBase;
}
