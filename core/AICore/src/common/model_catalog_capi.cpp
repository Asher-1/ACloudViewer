// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/model_catalog_capi.h"

#include <array>
#include <cstddef>
#include <cstring>

#include "aicore/asset_digests.h"

namespace {

using Entry = aicore_model_entry;

constexpr Entry kDepthModels[] = {
        {"depth-anything-base-q8_0.gguf", "Base Q8_0 (recommended)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-base-q8_0.gguf",
         "depth", 0, nullptr},
        {"depth-anything-base-q4_k.gguf", "Base Q4_K (smallest)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-base-q4_k.gguf",
         "depth", 0, nullptr},
        {"depth-anything-base-f16.gguf", "Base F16 (half precision)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-base-f16.gguf",
         "depth", 0, nullptr},
        {"depth-anything-large-q8_0.gguf", "Large Q8_0 (better quality)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-large-q8_0.gguf",
         "depth", 0, nullptr},
        {"depth-anything-large-q4_k.gguf", "Large Q4_K (compact)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-large-q4_k.gguf",
         "depth", 0, nullptr},
        {"depth-anything-giant-q8_0.gguf", "Giant Q8_0 (best quality)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-giant-q8_0.gguf",
         "depth", 0, nullptr},
        {"depth-anything-giant-q4_k.gguf", "Giant Q4_K (balanced)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-giant-q4_k.gguf",
         "depth", 0, nullptr},
        {"depth-anything-nested-metric.gguf", "Nested Metric F32",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-nested-metric.gguf",
         "metric", 0, nullptr},
        {"depth-anything-nested-anyview-q8_0.gguf", "Nested AnyView Q8_0",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-nested-anyview-q8_0.gguf",
         "metric", 0, nullptr},
        {"depth-anything-nested-anyview-q4_k.gguf", "Nested AnyView Q4_K",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DA3/depth-anything-nested-anyview-q4_k.gguf",
         "metric", 0, nullptr},
};

constexpr Entry kDeepLSDModels[] = {
        {"deeplsd_wireframe-f16.gguf", "DeepLSD Wireframe F16 (recommended)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DeepLSD/deeplsd_wireframe-f16.gguf",
         "", 0, nullptr},
        {"deeplsd_wireframe-q8_0.gguf", "DeepLSD Wireframe Q8_0 (smaller)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DeepLSD/deeplsd_wireframe-q8_0.gguf",
         "", 0, nullptr},
        {"deeplsd_wireframe-f32.gguf", "DeepLSD Wireframe F32",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DeepLSD/deeplsd_wireframe-f32.gguf",
         "", 0, nullptr},
        {"deeplsd_md-f16.gguf", "DeepLSD MegaDepth F16 (outdoor)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DeepLSD/deeplsd_md-f16.gguf",
         "", 0, nullptr},
        {"deeplsd_md-q8_0.gguf", "DeepLSD MegaDepth Q8_0",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DeepLSD/deeplsd_md-q8_0.gguf",
         "", 0, nullptr},
        {"deeplsd_md-f32.gguf", "DeepLSD MegaDepth F32",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "DeepLSD/deeplsd_md-f32.gguf",
         "", 0, nullptr},
};

constexpr Entry kLightGlueModels[] = {
        {"sift-lightglue-f16.gguf", "SIFT F16 (recommended)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/sift-lightglue-f16.gguf",
         "sift", 1, nullptr},
        {"sift-lightglue-q8_0.gguf", "SIFT Q8_0 (smaller)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/sift-lightglue-q8_0.gguf",
         "sift", 1, nullptr},
        {"sift-lightglue-f32.gguf", "SIFT F32",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/sift-lightglue-f32.gguf",
         "sift", 1, nullptr},
        {"aliked-lightglue-f16.gguf", "ALIKED F16 (recommended)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/aliked-lightglue-f16.gguf",
         "aliked", 2, nullptr},
        {"aliked-lightglue-q8_0.gguf", "ALIKED Q8_0 (smaller)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/aliked-lightglue-q8_0.gguf",
         "aliked", 2, nullptr},
        {"aliked-lightglue-f32.gguf", "ALIKED F32",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/aliked-lightglue-f32.gguf",
         "aliked", 2, nullptr},
};

constexpr Entry kAlikedModels[] = {
        {"aliked-n16rot-f16.gguf", "ALIKED N16Rot F16",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/aliked-n16rot-f16.gguf",
         "extractor", 2, nullptr},
        {"aliked-n16rot-q8_0.gguf", "ALIKED N16Rot Q8_0",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/aliked-n16rot-q8_0.gguf",
         "extractor", 2, nullptr},
        {"aliked-n16rot-f32.gguf", "ALIKED N16Rot F32",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "LightGlue/aliked-n16rot-f32.gguf",
         "extractor", 2, nullptr},
};

constexpr Entry kGaussianModels[] = {
        {"freesplatter-scene-q8_0.gguf", "Scene Q8_0 (recommended)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-scene-q8_0.gguf",
         "scene", 0, nullptr},
        {"freesplatter-scene-f16.gguf", "Scene F16",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-scene-f16.gguf",
         "scene", 0, nullptr},
        {"freesplatter-scene-f32.gguf", "Scene F32 (full precision)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-scene-f32.gguf",
         "scene", 0, nullptr},
        {"freesplatter-object-2dgs-q8_0.gguf", "Object-2DGS Q8_0 (recommended)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-object-2dgs-q8_0.gguf",
         "object", 0, nullptr},
        {"freesplatter-object-2dgs-f16.gguf", "Object-2DGS F16",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-object-2dgs-f16.gguf",
         "object", 0, nullptr},
        {"freesplatter-object-2dgs-f32.gguf",
         "Object-2DGS F32 (full precision)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-object-2dgs-f32.gguf",
         "object", 0, nullptr},
        {"freesplatter-object-q8_0.gguf", "Object-3DGS Q8_0 (deprecated)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-object-q8_0.gguf",
         "object", 0, nullptr},
        {"freesplatter-object-f16.gguf", "Object-3DGS F16 (deprecated)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-object-f16.gguf",
         "object", 0, nullptr},
        {"freesplatter-object-f32.gguf", "Object-3DGS F32 (deprecated)",
         "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
         "3dgs/freesplatter-object-f32.gguf",
         "object", 0, nullptr},
};

template <size_t N>
class Catalog {
public:
    explicit Catalog(const Entry (&source)[N]) {
        for (size_t i = 0; i < N; ++i) {
            entries_[i] = source[i];
            entries_[i].sha256 = aicore::AssetDigestForFile(source[i].filename);
        }
    }

    int size() const { return static_cast<int>(N); }

    const Entry* at(int index) const {
        return index >= 0 && index < size() ? &entries_[index] : nullptr;
    }

private:
    std::array<Entry, N> entries_{};
};

const auto& depthCatalog() {
    static const Catalog catalog(kDepthModels);
    return catalog;
}

const auto& deepLsdCatalog() {
    static const Catalog catalog(kDeepLSDModels);
    return catalog;
}

const auto& lightGlueCatalog() {
    static const Catalog catalog(kLightGlueModels);
    return catalog;
}

const auto& gaussianCatalog() {
    static const Catalog catalog(kGaussianModels);
    return catalog;
}

const auto& alikedCatalog() {
    static const Catalog catalog(kAlikedModels);
    return catalog;
}

}  // namespace

extern "C" {

AICORE_CAPI int aicore_model_count(aicore_model_family family) {
    switch (family) {
        case AICORE_MODEL_FAMILY_DEPTH:
            return depthCatalog().size();
        case AICORE_MODEL_FAMILY_DEEPLSD:
            return deepLsdCatalog().size();
        case AICORE_MODEL_FAMILY_LIGHTGLUE:
            return lightGlueCatalog().size();
        case AICORE_MODEL_FAMILY_GAUSSIAN:
            return gaussianCatalog().size();
        case AICORE_MODEL_FAMILY_ALIKED:
            return alikedCatalog().size();
    }
    return 0;
}

AICORE_CAPI const aicore_model_entry* aicore_model_at(
        aicore_model_family family, int index) {
    switch (family) {
        case AICORE_MODEL_FAMILY_DEPTH:
            return depthCatalog().at(index);
        case AICORE_MODEL_FAMILY_DEEPLSD:
            return deepLsdCatalog().at(index);
        case AICORE_MODEL_FAMILY_LIGHTGLUE:
            return lightGlueCatalog().at(index);
        case AICORE_MODEL_FAMILY_GAUSSIAN:
            return gaussianCatalog().at(index);
        case AICORE_MODEL_FAMILY_ALIKED:
            return alikedCatalog().at(index);
    }
    return nullptr;
}

AICORE_CAPI const aicore_model_entry* aicore_model_by_filename(
        aicore_model_family family, const char* filename) {
    if (!filename || !*filename) return nullptr;
    const int count = aicore_model_count(family);
    for (int i = 0; i < count; ++i) {
        const aicore_model_entry* entry = aicore_model_at(family, i);
        if (entry && entry->filename &&
            std::strcmp(entry->filename, filename) == 0) {
            return entry;
        }
    }
    return nullptr;
}

}  // extern "C"
