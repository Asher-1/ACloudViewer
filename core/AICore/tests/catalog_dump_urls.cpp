// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Dump every supported model asset. The default one-URL-per-line format feeds
// check_catalog_assets.py. --json emits the cache destination and pinned digest
// consumed by scripts/validate_all.py.

#include <cstdio>
#include <cstring>
#include <string>

#include "aicore/asset_digests.h"
#include "aicore/facedetect_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3_capi.h"
#include "aicore/trellis_capi.h"
#include "aicore/yolo_capi.h"

namespace {

bool g_json = false;
bool g_complete = true;

std::string jsonEscape(const char* text) {
    std::string result;
    for (const unsigned char ch : std::string(text ? text : "")) {
        switch (ch) {
            case '\\':
                result += "\\\\";
                break;
            case '"':
                result += "\\\"";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            case '\t':
                result += "\\t";
                break;
            default:
                if (ch < 0x20) {
                    char escaped[7]{};
                    std::snprintf(escaped, sizeof(escaped), "\\u%04x", ch);
                    result += escaped;
                } else {
                    result += static_cast<char>(ch);
                }
        }
    }
    return result;
}

void emitAsset(const char* task,
               const char* folder,
               const char* filename,
               const char* url,
               long long size_bytes = 0) {
    if (filename == nullptr || url == nullptr) return;
    if (!g_json) {
        std::printf("%s\n", url);
        return;
    }
    const char* digest = aicore::AssetDigestForFile(filename);
    if (digest == nullptr) g_complete = false;
    const std::string relative = std::string(folder) + "/" + filename;
    std::printf(
            "{\"task\":\"%s\",\"relative_path\":\"%s\","
            "\"url\":\"%s\",\"sha256\":\"%s\",\"size_bytes\":%lld}\n",
            jsonEscape(task).c_str(), jsonEscape(relative.c_str()).c_str(),
            jsonEscape(url).c_str(), jsonEscape(digest).c_str(), size_bytes);
}

template <size_t N>
void emitFixed(const char* task,
               const char* folder,
               const char* release_tag,
               const char* const (&filenames)[N]) {
    const std::string base =
            "https://github.com/Asher-1/cloudViewer_downloads/releases/"
            "download/" +
            std::string(release_tag) + "/";
    for (const char* filename : filenames) {
        const std::string url = base + filename;
        emitAsset(task, folder, filename, url.c_str());
    }
}

void emitFixedCatalogs() {
    static constexpr const char* kDepth[] = {
            "depth-anything-base-q8_0.gguf",
            "depth-anything-base-q4_k.gguf",
            "depth-anything-base-f16.gguf",
            "depth-anything-large-q8_0.gguf",
            "depth-anything-large-q4_k.gguf",
            "depth-anything-giant-q8_0.gguf",
            "depth-anything-giant-q4_k.gguf",
            "depth-anything-nested-metric.gguf",
            "depth-anything-nested-anyview-q8_0.gguf",
            "depth-anything-nested-anyview-q4_k.gguf",
    };
    static constexpr const char* kGaussian[] = {
            "freesplatter-scene-q8_0.gguf",
            "freesplatter-scene-f16.gguf",
            "freesplatter-scene-f32.gguf",
            "freesplatter-object-2dgs-q8_0.gguf",
            "freesplatter-object-2dgs-f16.gguf",
            "freesplatter-object-2dgs-f32.gguf",
            "freesplatter-object-q8_0.gguf",
            "freesplatter-object-f16.gguf",
            "freesplatter-object-f32.gguf",
    };
    static constexpr const char* kAliked[] = {
            "aliked-n16rot-f16.gguf",
            "aliked-n16rot-q8_0.gguf",
            "aliked-n16rot-f32.gguf",
    };
    static constexpr const char* kLightGlue[] = {
            "sift-lightglue-f16.gguf",    "sift-lightglue-q8_0.gguf",
            "sift-lightglue-f32.gguf",    "aliked-lightglue-f16.gguf",
            "aliked-lightglue-q8_0.gguf", "aliked-lightglue-f32.gguf",
    };
    static constexpr const char* kDeepLsd[] = {
            "deeplsd_wireframe-f16.gguf", "deeplsd_wireframe-q8_0.gguf",
            "deeplsd_wireframe-f32.gguf", "deeplsd_md-f16.gguf",
            "deeplsd_md-q8_0.gguf",       "deeplsd_md-f32.gguf",
    };
    emitFixed("depth", "da3_models", "DA3", kDepth);
    emitFixed("gaussian", "freesplatter_models", "3dgs", kGaussian);
    emitFixed("aliked", "lightglue_models", "LightGlue", kAliked);
    emitFixed("lightglue", "lightglue_models", "LightGlue", kLightGlue);
    emitFixed("deeplsd", "deeplsd_models", "DeepLSD", kDeepLsd);
}

void emitRuntimeCatalogs() {
    for (int i = 0; i < aicore_facedetect_model_count(); ++i) {
        const aicore_facedetect_model_entry* e = aicore_facedetect_model_at(i);
        if (e)
            emitAsset("facedetect", "facedetect_models", e->filename,
                      e->download_url);
    }
    for (int i = 0; i < aicore_rfdetr_model_count(); ++i) {
        const aicore_rfdetr_model_entry* e = aicore_rfdetr_model_at(i);
        if (e)
            emitAsset("rfdetr", "rfdetr_models", e->filename, e->download_url);
    }
    for (int i = 0; i < aicore_rmbg_model_count(); ++i) {
        const aicore_rmbg_model_entry* e = aicore_rmbg_model_at(i);
        if (e) emitAsset("rmbg", "rmbg_models", e->filename, e->download_url);
    }
    for (int i = 0; i < aicore_sam3_model_count(); ++i) {
        const aicore_sam3_model_entry* e = aicore_sam3_model_at(i);
        if (e)
            emitAsset("sam3", "sam3_models", e->filename, e->download_url,
                      static_cast<long long>(e->size_bytes));
    }
    for (int i = 0; i < aicore_trellis_model_count(); ++i) {
        const aicore_trellis_model_entry* e = aicore_trellis_model_at(i);
        // RMBG is a Trellis runtime dependency but its one physical cache and
        // regression ownership belong to the RMBG task above.
        if (e && !(e->role && std::strcmp(e->role, "rmbg") == 0))
            emitAsset("trellis", "trellis_models", e->filename, e->download_url,
                      static_cast<long long>(e->size_bytes));
    }
    const int yolo_count = aicore_yolo_model_count(AICORE_YOLO_ROLE_ANY);
    for (int i = 0; i < yolo_count; ++i) {
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_ANY);
        if (e) emitAsset("yolo", "yolo_models", e->filename, e->download_url);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--json") != 0)) {
        std::fprintf(stderr, "usage: %s [--json]\n", argv[0]);
        return 2;
    }
    g_json = argc == 2;
    emitFixedCatalogs();
    emitRuntimeCatalogs();
    if (!g_complete) {
        std::fprintf(
                stderr,
                "catalog contains assets without pinned SHA-256 digests\n");
        return 1;
    }
    return 0;
}
