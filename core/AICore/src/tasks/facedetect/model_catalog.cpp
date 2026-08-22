// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>

#include "aicore/facedetect_capi.h"

namespace {

static constexpr const char* kDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "qFaceDetect/";

struct ModelRow {
    const char* filename;
    const char* download_url;
    const char* display_name;
    const char* quant_note;
    const char* license_note;
    int detector_capable;
};

static constexpr ModelRow kModels[] = {
        {"buffalo_l.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "qFaceDetect/buffalo_l.gguf",
         "Buffalo L (recommended)",
         "F16 — primary pack (SCRFD + ArcFace 512-d)",
         "Non-commercial (insightface)", 1},
        {"buffalo_m.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "qFaceDetect/buffalo_m.gguf",
         "Buffalo M", "F16 — medium SCRFD + ArcFace 512-d",
         "Non-commercial (insightface)", 1},
        {"buffalo_s.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "qFaceDetect/buffalo_s.gguf",
         "Buffalo S", "F16 — smallest buffalo pack",
         "Non-commercial (insightface)", 1},
        {"buffalo_sc.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "qFaceDetect/buffalo_sc.gguf",
         "Buffalo SC", "F16 — compact det_500m + MobileFaceNet",
         "Non-commercial (insightface)", 1},
        {"antelopev2.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "qFaceDetect/antelopev2.gguf",
         "AntelopeV2", "F16 — SCRFD-10G + ArcFace R100 (highest accuracy)",
         "Non-commercial (insightface)", 1},
        {"yunet-sface.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "qFaceDetect/yunet-sface.gguf",
         "YuNet + SFace (commercial-friendly)",
         "F16 — YuNet detector + SFace 128-d (Apache-2.0)",
         "Apache-2.0 — commercial use OK", 1},
        {"landmarks-2d106-1k3d68.gguf",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "qFaceDetect/landmarks-2d106-1k3d68.gguf",
         "Dense landmarks only",
         "F16 — 106-pt 2D + 68-pt 3D heads (needs separate detector)",
         "Non-commercial; not for live detect", 0},
};

static aicore_facedetect_model_entry to_entry(const ModelRow& row) {
    return {row.filename,   row.download_url, row.display_name,
            row.quant_note, row.license_note, row.detector_capable};
}

static int detector_index_map(int detector_index) {
    int seen = -1;
    for (size_t i = 0; i < sizeof(kModels) / sizeof(kModels[0]); ++i) {
        if (!kModels[i].detector_capable) continue;
        ++seen;
        if (seen == detector_index) return static_cast<int>(i);
    }
    return -1;
}

static int landmark_index_map(int landmark_index) {
    int seen = -1;
    for (size_t i = 0; i < sizeof(kModels) / sizeof(kModels[0]); ++i) {
        if (kModels[i].detector_capable) continue;
        ++seen;
        if (seen == landmark_index) return static_cast<int>(i);
    }
    return -1;
}

}  // namespace

AICORE_CAPI int aicore_facedetect_model_count(void) {
    return static_cast<int>(sizeof(kModels) / sizeof(kModels[0]));
}

AICORE_CAPI const aicore_facedetect_model_entry* aicore_facedetect_model_at(
        int index) {
    static thread_local aicore_facedetect_model_entry entry{};
    if (index < 0 ||
        index >= static_cast<int>(sizeof(kModels) / sizeof(kModels[0]))) {
        return nullptr;
    }
    entry = to_entry(kModels[static_cast<size_t>(index)]);
    return &entry;
}

AICORE_CAPI int aicore_facedetect_detector_model_count(void) {
    int n = 0;
    for (const auto& row : kModels) {
        if (row.detector_capable) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_detector_model_at(int index) {
    const int mapped = detector_index_map(index);
    return mapped < 0 ? nullptr : aicore_facedetect_model_at(mapped);
}

AICORE_CAPI int aicore_facedetect_landmark_model_count(void) {
    int n = 0;
    for (const auto& row : kModels) {
        if (!row.detector_capable) ++n;
    }
    return n;
}

AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_landmark_model_at(int index) {
    const int mapped = landmark_index_map(index);
    return mapped < 0 ? nullptr : aicore_facedetect_model_at(mapped);
}

AICORE_CAPI const aicore_facedetect_model_entry*
aicore_facedetect_model_by_filename(const char* filename) {
    if (filename == nullptr || filename[0] == '\0') return nullptr;
    for (size_t i = 0; i < sizeof(kModels) / sizeof(kModels[0]); ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) {
            return aicore_facedetect_model_at(static_cast<int>(i));
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_facedetect_model_download_base(void) {
    return kDownloadBase;
}
