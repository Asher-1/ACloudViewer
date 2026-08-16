// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>

#include "aicore/rmbg_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "trellis2-ggml/";

struct ModelRow {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
};

// Unified GGUF (encoder + decoder in one file) converted from the official
// BRIA RMBG-2.0 weights (BiRefNet-Swin-L backbone).
static constexpr ModelRow kModels[] = {
        {"rmbg_f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "trellis2-ggml/rmbg_f16.gguf",
         "RMBG-2.0 F16 (recommended)",
         "F16 — unified BiRefNet-Swin-L encoder + decoder (~450 MB)",
         "CC BY-NC 4.0 (non-commercial); commercial license from BRIA"},
};

static aicore_rmbg_model_entry to_entry(const ModelRow& row) {
    return {row.filename, row.download_url, row.display_name, row.quant_note,
            row.license_note};
}

}  // namespace

AICORE_CAPI int aicore_rmbg_model_count(void) {
    return static_cast<int>(sizeof(kModels) / sizeof(kModels[0]));
}

AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_at(int index) {
    static thread_local aicore_rmbg_model_entry entry{};
    if (index < 0 ||
        index >= static_cast<int>(sizeof(kModels) / sizeof(kModels[0]))) {
        return nullptr;
    }
    entry = to_entry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < sizeof(kModels) / sizeof(kModels[0]); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_rmbg_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_rmbg_model_download_base(void) {
    return kDownloadBase;
}
