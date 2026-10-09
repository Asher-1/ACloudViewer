// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Published MapAnything-family GGUF catalog. Source of truth: the Hugging
// Face repo Asher-1/map_anything_vggt (exact LFS sizes + SHA-256 pinned in
// aicore/asset_digests.h). The regression runner discovers models only
// through this catalog - never through local file listings.
//
// Row order is the qMapAnything UI contract: the model-variant grouping
// (column 6) lists vggt-omega first (the accuracy pick of the published
// family) and dust3r last (the throughput pick); the first variant is the
// dialog's first tab. The catalog default row is the vggt-omega-512 Q8_0
// checkpoint (its quant note carries the visible "(recommended)" marker).

#include <array>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"

namespace {

// Hugging Face resolve-main base (ecvModelDownloader follows the 302 to the
// CDN automatically; public repo, no token needed).
static constexpr const char* kDownloadBase =
        "https://huggingface.co/Asher-1/map_anything_vggt/resolve/main/";

struct ModelRow {
    const char* filename;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    const char* variant;
    int64_t size_bytes;
    // Official-parity inference defaults (docs in the public header).
    const char* default_resize_mode;
    const char* conf_filter;
    float default_min_conf;
};

static constexpr ModelRow kModels[] = {
        {"vggt-omega-1b-512-f16.gguf", "VGGT-Omega 1B 512",
         "F16 - near-lossless reference",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-512", 2290343200, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-512-q8_0.gguf", "VGGT-Omega 1B 512",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-512", 1316485984, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-512-q6_K.gguf", "VGGT-Omega 1B 512",
         "Q6_K - high-precision compact, better than Q5_K",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-512", 1047802656, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-512-q5_K.gguf", "VGGT-Omega 1B 512", "Q5_K - compact",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-512", 900460096, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-512-f32.gguf", "VGGT-Omega 1B 512",
         "F32 - full precision reference",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-512", 4575356192, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-416-reproduce-f16.gguf",
         "VGGT-Omega 1B 416 (reproduce)", "F16 - near-lossless reference",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-416-reproduce", 2290343296, "balanced", "normalized",
         0.67f},
        {"vggt-omega-1b-416-reproduce-q8_0.gguf",
         "VGGT-Omega 1B 416 (reproduce)",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-416-reproduce", 1316486080, "balanced", "normalized",
         0.67f},
        {"vggt-omega-1b-416-reproduce-q6_K.gguf",
         "VGGT-Omega 1B 416 (reproduce)",
         "Q6_K - high-precision compact, better than Q5_K",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-416-reproduce", 1047802656, "balanced", "normalized",
         0.67f},
        {"vggt-omega-1b-416-reproduce-q5_K.gguf",
         "VGGT-Omega 1B 416 (reproduce)", "Q5_K - compact",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-416-reproduce", 900460096, "balanced", "normalized",
         0.67f},
        {"vggt-omega-1b-416-reproduce-f32.gguf",
         "VGGT-Omega 1B 416 (reproduce)", "F32 - full precision reference",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-416-reproduce", 4575356288, "balanced", "normalized",
         0.67f},
        {"vggt-omega-1b-256-text-f16.gguf", "VGGT-Omega 1B 256 (text)",
         "F16 - near-lossless reference",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-256-text", 2701944928, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-256-text-q8_0.gguf", "VGGT-Omega 1B 256 (text)",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-256-text", 1535411872, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-256-text-q6_K.gguf", "VGGT-Omega 1B 256 (text)",
         "Q6_K - high-precision compact, better than Q5_K",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-256-text", 1216953856, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-256-text-q5_K.gguf", "VGGT-Omega 1B 256 (text)",
         "Q5_K - compact",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-256-text", 1042315552, "balanced", "normalized", 0.67f},
        {"vggt-omega-1b-256-text-f32.gguf", "VGGT-Omega 1B 256 (text)",
         "F32 - full precision reference",
         "VGGT-Omega lineage: verify the upstream VGGT license before "
         "commercial use",
         "vggt-omega-1b-256-text", 5397999712, "balanced", "normalized", 0.67f},
        {"mapanything-f16.gguf", "MapAnything", "F16 - near-lossless reference",
         "Apache-2.0 (upstream MapAnything)", "mapanything", 2950534656,
         "balanced", "percentile", 3.0f},
        {"mapanything-q8_0.gguf", "MapAnything",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "Apache-2.0 (upstream MapAnything)", "mapanything", 1324961824,
         "balanced", "percentile", 3.0f},
        {"mapanything-q6_K.gguf", "MapAnything",
         "Q6_K - high-precision compact, better than Q5_K",
         "Apache-2.0 (upstream MapAnything)", "mapanything", 1050620768,
         "balanced", "percentile", 3.0f},
        {"mapanything-q5_K.gguf", "MapAnything", "Q5_K - compact",
         "Apache-2.0 (upstream MapAnything)", "mapanything", 900175648,
         "balanced", "percentile", 3.0f},
        {"mapanything-f32.gguf", "MapAnything",
         "F32 - full precision reference", "Apache-2.0 (upstream MapAnything)",
         "mapanything", 4652453536, "balanced", "percentile", 3.0f},
        {"pi3x-f16.gguf", "Pi3X", "F16 - near-lossless reference",
         "Apache-2.0 (upstream Pi3)", "pi3x", 2035329344, "balanced",
         "normalized", 0.5f},
        {"pi3x-q8_0.gguf", "Pi3X",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "Apache-2.0 (upstream Pi3)", "pi3x", 1128469184, "balanced",
         "normalized", 0.5f},
        {"pi3x-q6_K.gguf", "Pi3X",
         "Q6_K - high-precision compact, better than Q5_K",
         "Apache-2.0 (upstream Pi3)", "pi3x", 894197024, "balanced",
         "normalized", 0.5f},
        {"pi3x-f32.gguf", "Pi3X", "F32 - full precision reference",
         "Apache-2.0 (upstream Pi3)", "pi3x", 3969964352, "balanced",
         "normalized", 0.5f},
        {"pi3-f16.gguf", "Pi3", "F16 - near-lossless reference",
         "Apache-2.0 (upstream Pi3)", "pi3", 1923798432, "balanced",
         "normalized", 0.5f},
        {"pi3-q8_0.gguf", "Pi3",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "Apache-2.0 (upstream Pi3)", "pi3", 1027982112, "balanced",
         "normalized", 0.5f},
        {"pi3-q5_K.gguf", "Pi3", "Q5_K - compact", "Apache-2.0 (upstream Pi3)",
         "pi3", 669655584, "balanced", "normalized", 0.5f},
        {"pi3-f32.gguf", "Pi3", "F32 - full precision reference",
         "Apache-2.0 (upstream Pi3)", "pi3", 3834873248, "balanced",
         "normalized", 0.5f},
        {"vggt-1b-f16.gguf", "VGGT-1B", "F16 - near-lossless reference",
         "VGGT lineage: verify the upstream VGGT license before commercial use",
         "vggt-1b", 2543647232, "square", "normalized", 0.8f},
        {"vggt-1b-q8_0.gguf", "VGGT-1B",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "VGGT lineage: verify the upstream VGGT license before commercial use",
         "vggt-1b", 1503619136, "square", "normalized", 0.8f},
        {"vggt-1b-q5_K.gguf", "VGGT-1B", "Q5_K - compact",
         "VGGT lineage: verify the upstream VGGT license before commercial use",
         "vggt-1b", 1087593152, "square", "normalized", 0.8f},
        {"vggt-1b-f32.gguf", "VGGT-1B", "F32 - full precision reference",
         "VGGT lineage: verify the upstream VGGT license before commercial use",
         "vggt-1b", 4762489344, "square", "normalized", 0.8f},
        {"dust3r-f16.gguf", "DUSt3R", "F16 - near-lossless reference",
         "CC-BY-NC (upstream DUSt3R); non-commercial", "dust3r", 1227788928,
         "balanced", "normalized", 0.5f},
        {"dust3r-q8_0.gguf", "DUSt3R",
         "Q8_0 - near-lossless, smallest fast default (recommended)",
         "CC-BY-NC (upstream DUSt3R); non-commercial", "dust3r", 732336768,
         "balanced", "normalized", 0.5f},
        {"dust3r-q6_K.gguf", "DUSt3R",
         "Q6_K - high-precision compact, better than Q5_K",
         "CC-BY-NC (upstream DUSt3R); non-commercial", "dust3r", 604344960,
         "balanced", "normalized", 0.5f},
        {"dust3r-f32.gguf", "DUSt3R", "F32 - full precision reference",
         "CC-BY-NC (upstream DUSt3R); non-commercial", "dust3r", 2284753536,
         "balanced", "normalized", 0.5f},
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

const aicore_map_anything_model_entry* catalog_rows() {
    static const std::vector<aicore_map_anything_model_entry> rows = [] {
        std::vector<aicore_map_anything_model_entry> v;
        v.reserve(kModelCount);
        const auto& urls = catalog_urls();
        for (int i = 0; i < kModelCount; ++i) {
            aicore_map_anything_model_entry e{};
            e.filename = kModels[i].filename;
            e.download_url = urls[i].c_str();
            e.display_name = kModels[i].display_name;
            e.quant_note = kModels[i].quant_note;
            e.license_note = kModels[i].license_note;
            e.variant = kModels[i].variant;
            e.size_bytes = kModels[i].size_bytes;
            e.default_resize_mode = kModels[i].default_resize_mode;
            e.conf_filter = kModels[i].conf_filter;
            e.default_min_conf = kModels[i].default_min_conf;
            v.push_back(e);
        }
        return v;
    }();
    return rows.data();
}

}  // namespace

AICORE_CAPI int aicore_map_anything_model_count(void) { return kModelCount; }

AICORE_CAPI const aicore_map_anything_model_entry* aicore_map_anything_model_at(
        int index) {
    if (index < 0 || index >= kModelCount) return nullptr;
    return catalog_rows() + index;
}

AICORE_CAPI int aicore_map_anything_model_default_index(void) {
    return default_index();
}

AICORE_CAPI const aicore_map_anything_model_entry*
aicore_map_anything_model_by_filename(const char* filename) {
    if (filename == nullptr) return nullptr;
    for (int i = 0; i < kModelCount; ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_map_anything_model_at(i);
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_map_anything_model_download_base(void) {
    return kDownloadBase;
}
