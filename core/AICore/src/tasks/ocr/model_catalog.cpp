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

// (release stem, user-facing role label, role, language, family label).
// Stems match the converter output; family prefixes the display name.
struct VariantInfo {
    const char* stem;
    const char* label;
    const char* role;
    const char* language;
    const char* family;
};

static constexpr const VariantInfo kVariants[] = {
        {"ppocr-det-ch", "Det", "det", "ch", "PP-OCRv4"},
        {"ppocr-rec-ch", "Rec", "rec", "ch", "PP-OCRv4"},
        {"ppocr-rec-en", "Rec", "rec", "en", "PP-OCRv4"},
        {"pplcnet-layout", "Layout", "layout", "cdla", "PP-OCRv4"},
        {"ppocr-cls", "Cls", "cls", "ch", "PP-OCRv4"},
        {"ppocrv4-source-det", "Det", "det", "ch", "ch_PP-OCRv4 (source)"},
        {"ppocrv4-source-rec", "Rec", "rec", "ch", "ch_PP-OCRv4 (source)"},
        {"ppocrv3-japan-rec", "Rec", "rec", "ja", "japan_PP-OCRv3"},
        {"ppocrv5-mobile-det", "Det", "det", "ch", "PP-OCRv5 Mobile"},
        {"ppocrv5-mobile-rec", "Rec", "rec", "ch", "PP-OCRv5 Mobile"},
        {"ppocrv6-tiny-det", "Det", "det", "ch", "PP-OCRv6 Tiny"},
        {"ppocrv6-tiny-rec", "Rec", "rec", "ch", "PP-OCRv6 Tiny"},
        {"ppocrv6-small-det", "Det", "det", "ch", "PP-OCRv6 Small"},
        {"ppocrv6-small-rec", "Rec", "rec", "ch", "PP-OCRv6 Small"},
        {"ppocrv6-medium-det", "Det", "det", "ch", "PP-OCRv6 Medium"},
        {"ppocrv6-medium-rec", "Rec", "rec", "ch", "PP-OCRv6 Medium"},
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
                    std::string(variant.family) + " " + variant.label +
                    (std::string(variant.language) == "ch"   ? " (Chinese) "
                     : std::string(variant.language) == "en" ? " (English) "
                     : std::string(variant.language) == "ja"
                             ? " (Japanese) "
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

// Exact source recipes at X-AnyLabeling-GGML 745f6c355f94. This list owns
// pipeline pairing/defaults, while kModels owns published download assets.
static constexpr aicore_ocr_preset_entry kPresets[] = {
        {"ppocr-v4-ch", "PP-OCRv4 (ch)", "ppocr-det-ch", "ppocr-rec-ch",
         "ppocr-cls", .3f, .6f, 1.5f, 1000, 960, 6, 6, .9f, .5f},
        {"ppocr-v4-en", "PP-OCRv4 (en)", "ppocr-det-ch", "ppocr-rec-en",
         "ppocr-cls", .3f, .6f, 1.5f, 1000, 960, 6, 6, .9f, .5f},
        {"ppocr-v4-source", "ch_PP-OCRv4", "ppocrv4-source-det",
         "ppocrv4-source-rec", "ppocr-cls", .3f, .6f, 1.5f, 1000, 960, 6, 6,
         .9f, .5f},
        {"ppocr-v3-japan", "japan_PP-OCRv3", "ppocrv4-source-det",
         "ppocrv3-japan-rec", "ppocr-cls", .3f, .6f, 1.5f, 1000, 960, 6, 6, .9f,
         .5f},
        {"ppocr-v5-mobile", "PP-OCRv5", "ppocrv5-mobile-det",
         "ppocrv5-mobile-rec", "ppocr-cls", .3f, .6f, 1.5f, 1000, 960, 6, 6,
         .9f, .5f},
        {"ppocr-v6-tiny", "PP-OCRv6 Tiny", "ppocrv6-tiny-det",
         "ppocrv6-tiny-rec", "ppocr-cls", .2f, .4f, 1.4f, 3000, 960, 6, 6, .9f,
         .5f},
        {"ppocr-v6-small", "PP-OCRv6 Small", "ppocrv6-small-det",
         "ppocrv6-small-rec", "ppocr-cls", .2f, .45f, 1.4f, 3000, 960, 6, 6,
         .9f, .5f},
        {"ppocr-v6-medium", "PP-OCRv6 Medium", "ppocrv6-medium-det",
         "ppocrv6-medium-rec", "ppocr-cls", .2f, .45f, 1.4f, 3000, 960, 6, 6,
         .9f, .5f},
};

static int modelCount() { return static_cast<int>(kModels.size()); }

static aicore_ocr_model_entry toEntry(const ModelRow& row) {
    return {row.filename, row.download_url, row.display_name, row.role,
            row.language};
}

}  // namespace

AICORE_CAPI int aicore_ocr_model_count(void) { return modelCount(); }

AICORE_CAPI int aicore_ocr_preset_count(void) {
    return sizeof(kPresets) / sizeof(kPresets[0]);
}

AICORE_CAPI const aicore_ocr_preset_entry* aicore_ocr_preset_at(int index) {
    return index >= 0 && index < aicore_ocr_preset_count() ? &kPresets[index]
                                                           : nullptr;
}

AICORE_CAPI const aicore_ocr_preset_entry* aicore_ocr_preset_by_rec_filename(
        const char* filename) {
    if (!filename) return nullptr;
    for (const auto& preset : kPresets)
        for (const char* quant : kQuantSuffixes)
            if (std::string(preset.rec_stem) + "-" + quant + ".gguf" ==
                filename)
                return &preset;
    return nullptr;
}

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
