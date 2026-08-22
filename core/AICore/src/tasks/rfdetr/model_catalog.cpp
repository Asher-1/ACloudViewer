// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>
#include <string>
#include <vector>

#include "aicore/rfdetr_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "RF-DETR-GGUF/";

// 11 variant names.
static constexpr const char* kVariantNames[] = {
        "nano",      "small",      "base",        "medium",
        "large",     "seg-nano",   "seg-small",   "seg-medium",
        "seg-large", "seg-xlarge", "seg-2xlarge",
};

static constexpr int kVariantSegStart = 5;  // index of first seg-* variant

static constexpr int kVariantCount =
        sizeof(kVariantNames) / sizeof(kVariantNames[0]);

// 4 quantization suffixes.
static constexpr const char* kQuantSuffixes[] = {"f32", "f16", "q8_0", "q4_K"};

static constexpr int kQuantCount =
        sizeof(kQuantSuffixes) / sizeof(kQuantSuffixes[0]);

// Descriptive quant notes.
static constexpr const char* kQuantNotes[] = {
        "F32 \xe2\x80\x94 full precision reference",
        "F16 \xe2\x80\x94 half precision (recommended)",
        "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade",
        "Q4_K \xe2\x80\x94 4-bit quant, smallest practical",
};

// Per-variant display name prefix.
static const char* variantDisplayName(int vi) {
    if (vi < kVariantSegStart) {
        static const char* detNames[] = {
                "RF-DETR Nano",   "RF-DETR Small", "RF-DETR Base",
                "RF-DETR Medium", "RF-DETR Large",
        };
        return (vi >= 0 && vi < 5) ? detNames[vi] : "?";
    } else {
        static const char* segNames[] = {
                "RF-DETR Seg-Nano",   "RF-DETR Seg-Small",
                "RF-DETR Seg-Medium", "RF-DETR Seg-Large",
                "RF-DETR Seg-XLarge", "RF-DETR Seg-2XLarge",
        };
        const int si = vi - kVariantSegStart;
        return (si >= 0 && si < 6) ? segNames[si] : "?";
    }
}

static int isSegmentationVariant(int vi) {
    return vi >= kVariantSegStart ? 1 : 0;
}

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
    int segmentation_capable;
};

// Build the flat model list at init time.
static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    rows.reserve(kVariantCount * kQuantCount);
    for (int vi = 0; vi < kVariantCount; ++vi) {
        for (int qi = 0; qi < kQuantCount; ++qi) {
            std::string filename = std::string("rfdetr-") + kVariantNames[vi] +
                                   "-" + kQuantSuffixes[qi] + ".gguf";
            std::string url = std::string(kDownloadBase) + filename;
            std::string display = std::string(variantDisplayName(vi)) +
                                  " \xe2\x80\x94 " + kQuantNotes[qi];
            rows.push_back({dupString(filename.c_str()), dupString(url.c_str()),
                            dupString(display.c_str()),
                            dupString(kQuantNotes[qi]),
                            "Apache-2.0 (Roboflow RF-DETR)",
                            isSegmentationVariant(vi)});
        }
    }
    return rows;
}

static const std::vector<ModelRow> kModels = buildModels();

static int modelCount() { return static_cast<int>(kModels.size()); }

static int segIndexMap(int index) {
    int seen = -1;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (!kModels[i].segmentation_capable) continue;
        ++seen;
        if (seen == index) return static_cast<int>(i);
    }
    return -1;
}

static int detIndexMap(int index) {
    int seen = -1;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (kModels[i].segmentation_capable) continue;
        ++seen;
        if (seen == index) return static_cast<int>(i);
    }
    return -1;
}

static aicore_rfdetr_model_entry toEntry(const ModelRow& row) {
    return {row.filename,   row.download_url, row.display_name,
            row.quant_note, row.license_note, row.segmentation_capable};
}

}  // namespace

AICORE_CAPI int aicore_rfdetr_model_count(void) { return modelCount(); }

AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_model_at(int index) {
    static thread_local aicore_rfdetr_model_entry entry{};
    if (index < 0 || index >= modelCount()) return nullptr;
    entry = toEntry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI int aicore_rfdetr_detection_model_count(void) {
    int n = 0;
    for (const auto& row : kModels) {
        if (!row.segmentation_capable) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_detection_model_at(
        int index) {
    const int mapped = detIndexMap(index);
    return mapped < 0 ? nullptr : aicore_rfdetr_model_at(mapped);
}

AICORE_CAPI int aicore_rfdetr_segmentation_model_count(void) {
    int n = 0;
    for (const auto& row : kModels) {
        if (row.segmentation_capable) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_rfdetr_model_entry*
aicore_rfdetr_segmentation_model_at(int index) {
    const int mapped = segIndexMap(index);
    return mapped < 0 ? nullptr : aicore_rfdetr_model_at(mapped);
}

AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_rfdetr_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_rfdetr_model_download_base(void) {
    return kDownloadBase;
}