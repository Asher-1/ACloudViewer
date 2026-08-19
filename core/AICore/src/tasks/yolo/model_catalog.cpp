// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>
#include <string>
#include <vector>

#include "aicore/yolo_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "yolo_gguf_models/";

// 11 variants: yolov8 n/s/m/l/x, yolo26 n/s/m/l/x, yolo26n-depth. Filenames
// follow the yolo_gguf_models release exactly (33 assets, verified against
// the GitHub Release API — see docs: ultralytics-ggml-integration-plan.md).
static constexpr const char* kVariantNames[] = {
        "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",       "yolo26n",
        "yolo26s", "yolo26m", "yolo26l", "yolo26x", "yolo26n-depth",
};

static constexpr int kVariantCount =
        sizeof(kVariantNames) / sizeof(kVariantNames[0]);

// First yolo26* index (end2end head family).
static constexpr int kYolo26Start = 5;
// Depth variant index.
static constexpr int kDepthVariant = 10;

// 3 quantization suffixes.
static constexpr const char* kQuantSuffixes[] = {"f32", "f16", "q8_0"};

static constexpr int kQuantCount =
        sizeof(kQuantSuffixes) / sizeof(kQuantSuffixes[0]);

// Descriptive quant notes.
static constexpr const char* kQuantNotes[] = {
        "F32 \xe2\x80\x94 full precision reference",
        "F16 \xe2\x80\x94 half precision (recommended)",
        "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade",
};

static const char* variantDisplayName(int vi) {
    static const char* names[] = {
            "YOLOv8 Nano",   "YOLOv8 Small",      "YOLOv8 Medium",
            "YOLOv8 Large",  "YOLOv8 XLarge",     "YOLO26 Nano",
            "YOLO26 Small",  "YOLO26 Medium",     "YOLO26 Large",
            "YOLO26 XLarge", "YOLO26 Nano Depth",
    };
    return (vi >= 0 && vi < kVariantCount) ? names[vi] : "?";
}

static int isDepthVariant(int vi) { return vi == kDepthVariant ? 1 : 0; }

static int isEnd2EndVariant(int vi) {
    return vi >= kYolo26Start ? 1 : 0;  // yolo26 family (incl. depth)
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
    int depth_capable;
    int end2end;
};

// Build the flat model list at init time.
static std::vector<ModelRow> buildModels() {
    std::vector<ModelRow> rows;
    rows.reserve(kVariantCount * kQuantCount);
    for (int vi = 0; vi < kVariantCount; ++vi) {
        for (int qi = 0; qi < kQuantCount; ++qi) {
            std::string filename = std::string(kVariantNames[vi]) + "-" +
                                   kQuantSuffixes[qi] + ".gguf";
            std::string url = std::string(kDownloadBase) + filename;
            std::string display = std::string(variantDisplayName(vi)) +
                                  " \xe2\x80\x94 " + kQuantNotes[qi];
            rows.push_back({dupString(filename.c_str()), dupString(url.c_str()),
                            dupString(display.c_str()),
                            dupString(kQuantNotes[qi]),
                            "AGPL-3.0 (Ultralytics)", isDepthVariant(vi),
                            isEnd2EndVariant(vi)});
        }
    }
    return rows;
}

static const std::vector<ModelRow> kModels = buildModels();

static int modelCount() { return static_cast<int>(kModels.size()); }

static int depthIndexMap(int index) {
    int seen = -1;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (!kModels[i].depth_capable) continue;
        ++seen;
        if (seen == index) return static_cast<int>(i);
    }
    return -1;
}

static int detIndexMap(int index) {
    int seen = -1;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (kModels[i].depth_capable) continue;
        ++seen;
        if (seen == index) return static_cast<int>(i);
    }
    return -1;
}

static aicore_yolo_model_entry toEntry(const ModelRow& row) {
    return {row.filename,   row.download_url, row.display_name,
            row.quant_note, row.license_note, row.depth_capable,
            row.end2end};
}

}  // namespace

AICORE_CAPI int aicore_yolo_model_count(void) { return modelCount(); }

AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_at(int index) {
    static thread_local aicore_yolo_model_entry entry{};
    if (index < 0 || index >= modelCount()) return nullptr;
    entry = toEntry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI int aicore_yolo_detection_model_count(void) {
    int n = 0;
    for (const auto& row : kModels) {
        if (!row.depth_capable) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_detection_model_at(
        int index) {
    const int mapped = detIndexMap(index);
    return mapped < 0 ? nullptr : aicore_yolo_model_at(mapped);
}

AICORE_CAPI int aicore_yolo_depth_model_count(void) {
    int n = 0;
    for (const auto& row : kModels) {
        if (row.depth_capable) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_depth_model_at(
        int index) {
    const int mapped = depthIndexMap(index);
    return mapped < 0 ? nullptr : aicore_yolo_model_at(mapped);
}

AICORE_CAPI const aicore_yolo_model_entry* aicore_yolo_model_by_filename(
        const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < kModels.size(); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_yolo_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_yolo_model_download_base(void) {
    return kDownloadBase;
}
