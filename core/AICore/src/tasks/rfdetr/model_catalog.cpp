// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>

#include "aicore/rfdetr_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "RF-DETR-GGUF/";

struct ModelRow {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    int segmentation_capable;
};

// F16 unified GGUFs converted from the official Roboflow RF-DETR weights
// (same naming convention as the upstream mudler/rfdetr-cpp-* HuggingFace
// repositories). Detection variants use the COCO 80-class taxonomy; the Seg*
// variants add a mask head and run segmentation alongside detection.
static constexpr ModelRow kModels[] = {
        {"rfdetr-base-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-base-f16.gguf",
         "RF-DETR Base (recommended)",
         "F16 — balanced accuracy/speed (64 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 0},
        {"rfdetr-nano-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-nano-f16.gguf",
         "RF-DETR Nano",
         "F16 — fastest, edge-friendly (61 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 0},
        {"rfdetr-small-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-small-f16.gguf",
         "RF-DETR Small",
         "F16 — lightweight (64 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 0},
        {"rfdetr-medium-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-medium-f16.gguf",
         "RF-DETR Medium",
         "F16 — higher accuracy (67 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 0},
        {"rfdetr-large-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-large-f16.gguf",
         "RF-DETR Large",
         "F16 — highest detection accuracy (68 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 0},
        {"rfdetr-seg-nano-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-seg-nano-f16.gguf",
         "RF-DETR Seg-Nano",
         "F16 — detection + instance masks (68 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 1},
        {"rfdetr-seg-small-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-seg-small-f16.gguf",
         "RF-DETR Seg-Small",
         "F16 — detection + instance masks (68 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 1},
        {"rfdetr-seg-medium-f16.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "RF-DETR-GGUF/rfdetr-seg-medium-f16.gguf",
         "RF-DETR Seg-Medium",
         "F16 — detection + instance masks (72 MB)",
         "Apache-2.0 (Roboflow RF-DETR)", 1},
};

static aicore_rfdetr_model_entry to_entry(const ModelRow& row) {
    return {row.filename,   row.download_url, row.display_name,
            row.quant_note, row.license_note, row.segmentation_capable};
}

static int seg_index_map(int index) {
    int seen = -1;
    for (size_t i = 0; i < sizeof(kModels) / sizeof(kModels[0]); ++i) {
        if (!kModels[i].segmentation_capable) continue;
        ++seen;
        if (seen == index) return static_cast<int>(i);
    }
    return -1;
}

static int det_index_map(int index) {
    int seen = -1;
    for (size_t i = 0; i < sizeof(kModels) / sizeof(kModels[0]); ++i) {
        if (kModels[i].segmentation_capable) continue;
        ++seen;
        if (seen == index) return static_cast<int>(i);
    }
    return -1;
}

}  // namespace

AICORE_CAPI int aicore_rfdetr_model_count(void) {
    return static_cast<int>(sizeof(kModels) / sizeof(kModels[0]));
}

AICORE_CAPI const aicore_rfdetr_model_entry* aicore_rfdetr_model_at(
        int index) {
    static thread_local aicore_rfdetr_model_entry entry{};
    if (index < 0 ||
        index >= static_cast<int>(sizeof(kModels) / sizeof(kModels[0]))) {
        return nullptr;
    }
    entry = to_entry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI int aicore_rfdetr_detection_model_count(void) {
    int n = 0;
    for (const auto& row : kModels) {
        if (!row.segmentation_capable) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_rfdetr_model_entry*
aicore_rfdetr_detection_model_at(int index) {
    const int mapped = det_index_map(index);
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
    const int mapped = seg_index_map(index);
    return mapped < 0 ? nullptr : aicore_rfdetr_model_at(mapped);
}

AICORE_CAPI const aicore_rfdetr_model_entry*
aicore_rfdetr_model_by_filename(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < sizeof(kModels) / sizeof(kModels[0]); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_rfdetr_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_rfdetr_model_download_base(void) {
    return kDownloadBase;
}
