// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>
#include <string>
#include <vector>

#include "aicore/ocr_capi.h"
#include "common/capi_utils.hpp"
#include "common/model_cache.hpp"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "PP-OCR/";

// Quantization suffixes — these must match the assets actually published in
// the PP-OCR release (verified by test_catalog_remote_assets).
static constexpr const char* kQuantSuffixes[] = {"f32", "f16", "q8_0"};

static constexpr int kQuantCount =
        sizeof(kQuantSuffixes) / sizeof(kQuantSuffixes[0]);

static constexpr const char* kQuantNotes[] = {
        "F32 \xe2\x80\x94 full precision reference",
        "F16 \xe2\x80\x94 half precision (recommended)",
        "Q8_0 \xe2\x80\x94 8-bit block quant, best size trade"};

// (release stem, user-facing role label, role, language). Stems match
// the converter output.
struct VariantInfo {
    const char* stem;
    const char* label;
    const char* role;
    const char* language;
};

static constexpr const VariantInfo kVariants[] = {
        {"ppocr-det-ch", "Det", "det", "ch"},
        {"ppocr-rec-ch", "Rec", "rec", "ch"},
        {"ppocr-rec-en", "Rec", "rec", "en"},
        {"pplcnet-layout", "Layout", "layout", "cdla"},
};

static constexpr int kVariantCount = sizeof(kVariants) / sizeof(kVariants[0]);

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
    const char* language;
};

static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    rows.reserve(static_cast<size_t>(kVariantCount * kQuantCount));
    for (int vi = 0; vi < kVariantCount; ++vi) {
        const VariantInfo& variant = kVariants[vi];
        for (int qi = 0; qi < kQuantCount; ++qi) {
            std::string filename = std::string(variant.stem) + "-" +
                                   kQuantSuffixes[qi] + ".gguf";
            std::string url = std::string(kDownloadBase) + filename;
            std::string display =
                    std::string("PP-OCRv4 ") + variant.label +
                    (std::string(variant.language) == "ch" ? " (Chinese) "
                     : std::string(variant.language) == "en"
                             ? " (English) "
                             : " (CDLA document regions) ") +
                    kQuantNotes[qi];
            rows.push_back({dupString(filename.c_str()), dupString(url.c_str()),
                            dupString(display.c_str()), dupString(variant.role),
                            dupString(variant.language)});
        }
    }
    return rows;
}

static const std::vector<ModelRow> kModels = buildModels();

static int modelCount() { return static_cast<int>(kModels.size()); }

static aicore_ocr_model_entry toEntry(const ModelRow& row) {
    return {row.filename, row.download_url, row.display_name, row.role,
            row.language};
}

}  // namespace

AICORE_CAPI int aicore_ocr_model_count(void) { return modelCount(); }

AICORE_CAPI const aicore_ocr_model_entry* aicore_ocr_model_at(int index) {
    static thread_local aicore_ocr_model_entry entry{};
    if (index < 0 || index >= modelCount()) return nullptr;
    entry = toEntry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI const aicore_ocr_model_entry* aicore_ocr_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_ocr_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_ocr_model_download_base(void) {
    return kDownloadBase;
}

AICORE_CAPI char* aicore_ocr_model_cache_dir(void) {
    return aicore::capi::dup_cstr(aicore::ocr_model_cache_dir());
}

AICORE_CAPI void aicore_ocr_free_buffer(char* buffer) { free(buffer); }
