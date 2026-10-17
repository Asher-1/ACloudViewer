// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// `table` task catalog: SLANet_plus converted by
// tools/convert_paddle_ocr_gguf.py --role table, published in the PP-OCR
// release next to the det/rec/layout packs it composes with and cached in
// the same ocr_models folder (digests pinned in aicore/asset_digests.h).

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/table_capi.h"
#include "common/capi_utils.hpp"
#include "common/model_cache.hpp"

namespace {

constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "PP-OCR/";

struct Variant {
    const char* filename;
    const char* note;
};

constexpr Variant kVariants[] = {
        {"slanet-plus-table-f32.gguf",
         "F32 \xe2\x80\x94 full precision reference"},
        {"slanet-plus-table-f16.gguf",
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"slanet-plus-table-q8_0.gguf",
         "Q8_0 \xe2\x80\x94 8-bit block quant, best size trade"},
};

constexpr int kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

struct Row {
    std::string filename;
    std::string url;
    std::string display;
};

const std::vector<Row>& rows() {
    static const std::vector<Row> kRows = [] {
        std::vector<Row> out;
        for (const Variant& v : kVariants)
            out.push_back(
                    {v.filename, std::string(kDownloadBase) + v.filename,
                     std::string("SLANet_plus Table Structure ") + v.note});
        return out;
    }();
    return kRows;
}

}  // namespace

extern "C" {

int aicore_table_model_count(void) { return kVariantCount; }

const aicore_table_model_entry* aicore_table_model_at(int index) {
    static thread_local aicore_table_model_entry entry{};
    if (index < 0 || index >= kVariantCount) return nullptr;
    const Row& row = rows()[static_cast<size_t>(index)];
    entry = {row.filename.c_str(), row.url.c_str(), row.display.c_str(),
             "table", "slanet_plus"};
    return &entry;
}

const aicore_table_model_entry* aicore_table_model_by_filename(
        const char* filename) {
    if (!filename || !filename[0]) return nullptr;
    for (int i = 0; i < kVariantCount; ++i)
        if (std::strcmp(kVariants[i].filename, filename) == 0)
            return aicore_table_model_at(i);
    return nullptr;
}

const char* aicore_table_model_download_base(void) { return kDownloadBase; }

char* aicore_table_model_cache_dir(void) {
    return aicore::capi::dup_cstr(aicore::ocr_model_cache_dir());
}

void aicore_table_free_buffer(char* buffer) { std::free(buffer); }

}  // extern "C"
