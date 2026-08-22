// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>
#include <string>
#include <vector>

#include "aicore/rmbg_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "trellis2-ggml/";

// 3 quantization suffixes. These must match the assets actually published in
// the trellis2-ggml release (verified by the test_catalog_remote_assets test);
// rmbg_q8_0.gguf / rmbg_q4_K.gguf do not exist upstream.
static constexpr const char* kQuantSuffixes[] = {"f32", "f16", "q8"};

static constexpr int kQuantCount =
        sizeof(kQuantSuffixes) / sizeof(kQuantSuffixes[0]);

// Descriptive quant notes.
static constexpr const char* kQuantNotes[] = {
        "F32 \xe2\x80\x94 full precision reference",
        "F16 \xe2\x80\x94 half precision (recommended)",
        "Q8 \xe2\x80\x94 8-bit quant, best accuracy/size trade",
};

static constexpr const char* kLicenseNote =
        "CC BY-NC 4.0 (non-commercial); commercial license from BRIA";

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
};

// Build the model list at init time.
static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    rows.reserve(kQuantCount);
    for (int qi = 0; qi < kQuantCount; ++qi) {
        std::string filename =
                std::string("rmbg_") + kQuantSuffixes[qi] + ".gguf";
        std::string url = std::string(kDownloadBase) + filename;
        std::string display = std::string("RMBG-2.0 ") + kQuantNotes[qi];
        rows.push_back({dupString(filename.c_str()), dupString(url.c_str()),
                        dupString(display.c_str()), dupString(kQuantNotes[qi]),
                        dupString(kLicenseNote)});
    }
    return rows;
}

static const std::vector<ModelRow> kModels = buildModels();

static int modelCount() { return static_cast<int>(kModels.size()); }

static aicore_rmbg_model_entry toEntry(const ModelRow& row) {
    return {row.filename, row.download_url, row.display_name, row.quant_note,
            row.license_note};
}

}  // namespace

AICORE_CAPI int aicore_rmbg_model_count(void) { return modelCount(); }

AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_at(int index) {
    static thread_local aicore_rmbg_model_entry entry{};
    if (index < 0 || index >= modelCount()) return nullptr;
    entry = toEntry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI const aicore_rmbg_model_entry* aicore_rmbg_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_rmbg_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_rmbg_model_download_base(void) {
    return kDownloadBase;
}