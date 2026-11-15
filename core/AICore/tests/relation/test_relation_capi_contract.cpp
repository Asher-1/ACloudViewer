// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

#include "aicore/backend_capi.h"
#include "aicore/relation_capi.h"
#define REQUIRE(x)                                                      \
    do {                                                                \
        if (!(x)) {                                                     \
            std::fprintf(stderr, "failed line %d: %s\n", __LINE__, #x); \
            return 1;                                                   \
        }                                                               \
    } while (0)
int main() {
    auto o = aicore_relation_default_options();
    auto r = aicore_relation_default_request();
    REQUIRE(o.struct_size == sizeof(o) && r.struct_size == sizeof(r));
    REQUIRE(o.abi_version == 1 && r.abi_version == 1);
    REQUIRE(o.strict_f32 == -1 && o.flash_attention == 1 &&
            o.compact_pairs == 1 && o.fused_vocabulary == 1);
    aicore_model_device_info info{};
    info.struct_size = sizeof(info);
    REQUIRE(aicore_model_device_info_query(AICORE_MODEL_RELATION, "cpu",
                                           &info) == 0);
    REQUIRE(info.capabilities & AICORE_MODEL_CAP_FULL_GRAPH);
    REQUIRE(info.capabilities & AICORE_MODEL_CAP_TASK_CANCEL);
    auto* ctx = aicore_relation_create(nullptr);
    REQUIRE(ctx && !aicore_relation_ready(ctx));
    REQUIRE(std::strlen(aicore_relation_last_error(ctx)));
    aicore_relation_free(ctx);
    o.struct_size = 1;
    ctx = aicore_relation_create(&o);
    REQUIRE(!aicore_relation_ready(ctx));
    aicore_relation_free(ctx);
    o = aicore_relation_default_options();
    o.abi_version = 99;
    ctx = aicore_relation_create(&o);
    REQUIRE(!aicore_relation_ready(ctx));
    aicore_relation_free(ctx);
    const int invalidModes[] = {-2, 2};
    for (int invalidMode : invalidModes) {
        o = aicore_relation_default_options();
        o.strict_f32 = invalidMode;
        ctx = aicore_relation_create(&o);
        REQUIRE(ctx && !aicore_relation_ready(ctx));
        REQUIRE(std::strstr(aicore_relation_last_error(ctx), "strict_f32"));
        aicore_relation_free(ctx);
    }
    aicore_relation_result* out = reinterpret_cast<aicore_relation_result*>(1);
    REQUIRE(aicore_relation_infer_image(nullptr, nullptr, &r, &out) < 0 &&
            !out);
    aicore_relation_embeddings* text =
            reinterpret_cast<aicore_relation_embeddings*>(1);
    REQUIRE(aicore_relation_encode_text(nullptr, nullptr, 0, &text) < 0 &&
            !text);
    aicore_pipeline_timings timings{};
    REQUIRE(aicore_relation_last_pipeline_timings(nullptr, &timings) < 0);
    REQUIRE(!aicore_relation_ready(nullptr) &&
            aicore_relation_last_error(nullptr));
    REQUIRE(aicore_relation_context_device(nullptr));
    float calibrationA = 7, calibrationB = 7;
    REQUIRE(aicore_relation_calibration(nullptr, &calibrationA, &calibrationB) <
            0);
    REQUIRE(aicore_relation_object_vocabulary_count() == 497);
    REQUIRE(std::strcmp(aicore_relation_object_vocabulary_at(0), "accordion") ==
                    0 &&
            std::strcmp(aicore_relation_object_vocabulary_at(496), "zebra") ==
                    0 &&
            !aicore_relation_object_vocabulary_at(-1) &&
            !aicore_relation_object_vocabulary_at(497));
    for (int i = 0; i < 497; ++i) {
        const char* n = aicore_relation_object_vocabulary_at(i);
        REQUIRE(n && n[0] && !std::strchr(n, ','));
    }
    // 9 ViT relation weights + 12 ConvNeXt relation weights, 3 text students
    // and the CLIP tokenizer.
    constexpr int kCatalogSize = 25;
    REQUIRE(aicore_relation_model_count() == kCatalogSize);
    REQUIRE(!aicore_relation_model_at(-1) &&
            !aicore_relation_model_at(kCatalogSize));
    int relations = 0, students = 0, tokenizers = 0, convnext = 0, vit = 0;
    int vitFiles = 0;
    std::set<std::string> filenames;
    for (int i = 0; i < kCatalogSize; ++i) {
        const auto* e = aicore_relation_model_at(i);
        REQUIRE(e && e->filename && e->download_url && e->architecture &&
                e->sha256);
        REQUIRE(e->size_bytes > 0 && std::strlen(e->sha256) == 64);
        REQUIRE(filenames.insert(e->filename).second);
        const std::string digest = e->sha256;
        for (char c : digest)
            REQUIRE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
        if (e->role == AICORE_RELATION_MODEL_RELATIONS) ++relations;
        if (e->role == AICORE_RELATION_MODEL_TEXT) ++students;
        if (e->role == AICORE_RELATION_MODEL_TOKENIZER) ++tokenizers;
        if (e->role == AICORE_RELATION_MODEL_RELATIONS &&
            std::strncmp(e->architecture, "convnext-", 9) == 0) {
            ++convnext;
            const std::string name = e->filename;
            const std::string url = e->download_url;
            // relsgg-convnext-{tiny,small,base,large}-{f16,f32,q8_0}.gguf
            REQUIRE(name.rfind("relsgg-convnext-", 0) == 0);
            REQUIRE(name.size() > 5 &&
                    name.compare(name.size() - 5, 5, ".gguf") == 0);
            REQUIRE(url.size() > name.size() + 1 &&
                    url.compare(url.size() - name.size() - 1, name.size() + 1,
                                "/" + name) == 0);
            // The architecture names the size encoded in the filename.
            REQUIRE(name.find(std::string(e->architecture)) !=
                    std::string::npos);
        } else if (e->role == AICORE_RELATION_MODEL_RELATIONS) {
            ++vit;
            const std::string name = e->filename;
            if (name.rfind("relsgg-vit", 0) == 0) ++vitFiles;
        }
    }
    REQUIRE(relations == 21 && students == 3 && tokenizers == 1);
    REQUIRE(convnext == 12 && vit == 9 && vitFiles == 9);
    REQUIRE(relations + students + tokenizers == kCatalogSize);
    const int defaultIndex = aicore_relation_model_default_index();
    REQUIRE(defaultIndex >= 0 && defaultIndex < kCatalogSize);
    const auto* defaultModel = aicore_relation_model_at(defaultIndex);
    REQUIRE(defaultModel->role == AICORE_RELATION_MODEL_RELATIONS);
    // The recommended default stays the ViT-S/16+ Q8_0 weight, not a
    // ConvNeXt entry (whose graph is ported separately).
    REQUIRE(std::strcmp(defaultModel->filename,
                        "relsgg-vits16plus-q8_0.gguf") == 0);
    REQUIRE(std::strncmp(defaultModel->architecture, "convnext-", 9) != 0);
    char* dir = aicore_relation_model_cache_dir();
    REQUIRE(dir && std::strstr(dir, "relation_models"));
    aicore_relation_free_buffer(dir);
    aicore_relation_free(nullptr);
    aicore_relation_result_free(nullptr);
    aicore_relation_embeddings_free(nullptr);
    aicore_relation_shutdown();
    aicore_relation_shutdown();
    return 0;
}
