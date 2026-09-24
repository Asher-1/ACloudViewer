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
#include "aicore/gkd_capi.h"
#include "aicore/lingbot_capi.h"
#include "aicore/loma_capi.h"
#include "aicore/model_catalog_capi.h"
#include "aicore/rfdetr_capi.h"
#include "aicore/rmbg_capi.h"
#include "aicore/sam3_capi.h"
#include "aicore/sam3d_capi.h"
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

void emitSharedCatalog(aicore_model_family family,
                       const char* task,
                       const char* folder) {
    for (int i = 0; i < aicore_model_count(family); ++i) {
        const aicore_model_entry* entry = aicore_model_at(family, i);
        if (entry) {
            emitAsset(task, folder, entry->filename, entry->download_url);
        }
    }
}

void emitFixedCatalogs() {
    emitSharedCatalog(AICORE_MODEL_FAMILY_DEPTH, "depth", "da3_models");
    emitSharedCatalog(AICORE_MODEL_FAMILY_GAUSSIAN, "gaussian",
                      "freesplatter_models");
    emitSharedCatalog(AICORE_MODEL_FAMILY_ALIKED, "aliked", "lightglue_models");
    emitSharedCatalog(AICORE_MODEL_FAMILY_LIGHTGLUE, "lightglue",
                      "lightglue_models");
    emitSharedCatalog(AICORE_MODEL_FAMILY_DEEPLSD, "deeplsd", "deeplsd_models");
}

void emitRuntimeCatalogs() {
    for (int i = 0; i < aicore_loma_model_count(); ++i) {
        const aicore_loma_model_entry* e = aicore_loma_model_at(i);
        if (e) emitAsset("loma", "loma_models", e->filename, e->download_url);
    }
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
    for (int i = 0; i < aicore_sam3d_model_count(); ++i) {
        const aicore_sam3d_model_entry* e = aicore_sam3d_model_at(i);
        if (e)
            emitAsset("sam3d", "sam3d_models", e->filename, e->download_url,
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
    // ReID consumes the classify-task YOLO GGUFs (aicore/reid_capi.h wraps
    // them as appearance encoders). These rows declare the consumption only:
    // url/digest/size must match the yolo rows above, and the physical cache
    // and regression ownership stay with the yolo task.
    const int reid_encoder_count =
            aicore_yolo_model_count(AICORE_YOLO_ROLE_CLASSIFY);
    for (int i = 0; i < reid_encoder_count; ++i) {
        const aicore_yolo_model_entry* e =
                aicore_yolo_model_at(i, AICORE_YOLO_ROLE_CLASSIFY);
        if (e) emitAsset("reid", "yolo_models", e->filename, e->download_url);
    }
    for (int i = 0; i < aicore_gkd_model_count(); ++i) {
        const aicore_gkd_model_entry* e = aicore_gkd_model_at(i);
        if (e)
            emitAsset("gkd", "gkd_models", e->filename, e->download_url,
                      static_cast<long long>(e->size_bytes));
    }
    for (int i = 0; i < aicore_lingbot_model_count(); ++i) {
        const aicore_lingbot_model_entry* e = aicore_lingbot_model_at(i);
        if (e)
            emitAsset("lingbot", "lingbot_models", e->filename, e->download_url,
                      static_cast<long long>(e->size_bytes));
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
