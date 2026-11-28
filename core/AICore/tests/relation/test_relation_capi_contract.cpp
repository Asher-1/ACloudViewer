// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>

#include "aicore/backend_capi.h"
#include "aicore/relation_capi.h"

TEST(RelationCApiContract, DefaultOptionsAndRequestAbi) {
    auto o = aicore_relation_default_options();
    auto r = aicore_relation_default_request();
    ASSERT_TRUE(o.struct_size == sizeof(o) && r.struct_size == sizeof(r));
    ASSERT_TRUE(o.abi_version == 1 && r.abi_version == 1);
    ASSERT_TRUE(o.strict_f32 == -1 && o.flash_attention == 1 &&
                o.compact_pairs == 1 && o.fused_vocabulary == 1);
}

TEST(RelationCApiContract, CpuSupportsFullGraphAndCancellation) {
    aicore_model_device_info info{};
    info.struct_size = sizeof(info);
    ASSERT_EQ(
            aicore_model_device_info_query(AICORE_MODEL_RELATION, "cpu", &info),
            0);
    ASSERT_TRUE(info.capabilities & AICORE_MODEL_CAP_FULL_GRAPH);
    ASSERT_TRUE(info.capabilities & AICORE_MODEL_CAP_TASK_CANCEL);
}

TEST(RelationCApiContract, InvalidCreatesExposeQueryableErrors) {
    auto o = aicore_relation_default_options();
    auto* ctx = aicore_relation_create(nullptr);
    ASSERT_TRUE(ctx && !aicore_relation_ready(ctx));
    ASSERT_TRUE(std::strlen(aicore_relation_last_error(ctx)));
    aicore_relation_free(ctx);
    o.struct_size = 1;
    ctx = aicore_relation_create(&o);
    ASSERT_FALSE(aicore_relation_ready(ctx));
    aicore_relation_free(ctx);
    o = aicore_relation_default_options();
    o.abi_version = 99;
    ctx = aicore_relation_create(&o);
    ASSERT_FALSE(aicore_relation_ready(ctx));
    aicore_relation_free(ctx);
    const int invalidModes[] = {-2, 2};
    for (int invalidMode : invalidModes) {
        o = aicore_relation_default_options();
        o.strict_f32 = invalidMode;
        ctx = aicore_relation_create(&o);
        ASSERT_TRUE(ctx && !aicore_relation_ready(ctx));
        ASSERT_TRUE(std::strstr(aicore_relation_last_error(ctx), "strict_f32"));
        aicore_relation_free(ctx);
    }
}

TEST(RelationCApiContract, NullInferenceClearsOutputsAndQueriesFail) {
    auto r = aicore_relation_default_request();
    aicore_relation_result* out = reinterpret_cast<aicore_relation_result*>(1);
    ASSERT_TRUE(aicore_relation_infer_image(nullptr, nullptr, &r, &out) < 0 &&
                !out);
    aicore_relation_embeddings* text =
            reinterpret_cast<aicore_relation_embeddings*>(1);
    ASSERT_TRUE(aicore_relation_encode_text(nullptr, nullptr, 0, &text) < 0 &&
                !text);
    aicore_pipeline_timings timings{};
    ASSERT_LT(aicore_relation_last_pipeline_timings(nullptr, &timings), 0);
    ASSERT_TRUE(!aicore_relation_ready(nullptr) &&
                aicore_relation_last_error(nullptr));
    ASSERT_TRUE(aicore_relation_context_device(nullptr));
    float calibrationA = 7, calibrationB = 7;
    ASSERT_LT(
            aicore_relation_calibration(nullptr, &calibrationA, &calibrationB),
            0);
}

TEST(RelationCApiContract, ObjectVocabularyHas497CleanEntries) {
    ASSERT_EQ(aicore_relation_object_vocabulary_count(), 497);
    ASSERT_TRUE(std::strcmp(aicore_relation_object_vocabulary_at(0),
                            "accordion") == 0 &&
                std::strcmp(aicore_relation_object_vocabulary_at(496),
                            "zebra") == 0 &&
                !aicore_relation_object_vocabulary_at(-1) &&
                !aicore_relation_object_vocabulary_at(497));
    for (int i = 0; i < 497; ++i) {
        const char* n = aicore_relation_object_vocabulary_at(i);
        ASSERT_TRUE(n && n[0] && !std::strchr(n, ','));
    }
}

TEST(RelationCApiContract, CatalogInventoryDigestsAndRecommendedDefault) {
    // 9 ViT relation weights + 12 ConvNeXt relation weights, 3 text students
    // and the CLIP tokenizer.
    constexpr int kCatalogSize = 25;
    ASSERT_EQ(aicore_relation_model_count(), kCatalogSize);
    ASSERT_TRUE(!aicore_relation_model_at(-1) &&
                !aicore_relation_model_at(kCatalogSize));
    int relations = 0, students = 0, tokenizers = 0, convnext = 0, vit = 0;
    int vitFiles = 0;
    std::set<std::string> filenames;
    for (int i = 0; i < kCatalogSize; ++i) {
        const auto* e = aicore_relation_model_at(i);
        ASSERT_TRUE(e && e->filename && e->download_url && e->architecture &&
                    e->sha256);
        ASSERT_TRUE(e->size_bytes > 0 && std::strlen(e->sha256) == 64);
        ASSERT_TRUE(filenames.insert(e->filename).second);
        const std::string digest = e->sha256;
        for (char c : digest)
            ASSERT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
        if (e->role == AICORE_RELATION_MODEL_RELATIONS) ++relations;
        if (e->role == AICORE_RELATION_MODEL_TEXT) ++students;
        if (e->role == AICORE_RELATION_MODEL_TOKENIZER) ++tokenizers;
        if (e->role == AICORE_RELATION_MODEL_RELATIONS &&
            std::strncmp(e->architecture, "convnext-", 9) == 0) {
            ++convnext;
            const std::string name = e->filename;
            const std::string url = e->download_url;
            // relsgg-convnext-{tiny,small,base,large}-{f16,f32,q8_0}.gguf
            ASSERT_EQ(name.rfind("relsgg-convnext-", 0), 0);
            ASSERT_TRUE(name.size() > 5 &&
                        name.compare(name.size() - 5, 5, ".gguf") == 0);
            ASSERT_TRUE(url.size() > name.size() + 1 &&
                        url.compare(url.size() - name.size() - 1,
                                    name.size() + 1, "/" + name) == 0);
            // The architecture names the size encoded in the filename.
            ASSERT_NE(name.find(std::string(e->architecture)),
                      std::string::npos);
        } else if (e->role == AICORE_RELATION_MODEL_RELATIONS) {
            ++vit;
            const std::string name = e->filename;
            if (name.rfind("relsgg-vit", 0) == 0) ++vitFiles;
        }
    }
    ASSERT_TRUE(relations == 21 && students == 3 && tokenizers == 1);
    ASSERT_TRUE(convnext == 12 && vit == 9 && vitFiles == 9);
    ASSERT_EQ(relations + students + tokenizers, kCatalogSize);
    const int defaultIndex = aicore_relation_model_default_index();
    ASSERT_TRUE(defaultIndex >= 0 && defaultIndex < kCatalogSize);
    const auto* defaultModel = aicore_relation_model_at(defaultIndex);
    ASSERT_EQ(defaultModel->role, AICORE_RELATION_MODEL_RELATIONS);
    // The recommended default stays the ViT-S/16+ Q8_0 weight, not a
    // ConvNeXt entry (whose graph is ported separately).
    ASSERT_EQ(
            std::strcmp(defaultModel->filename, "relsgg-vits16plus-q8_0.gguf"),
            0);
    ASSERT_NE(std::strncmp(defaultModel->architecture, "convnext-", 9), 0);
}

TEST(RelationCApiContract, OwnedCacheNullReleaseAndIdempotentShutdown) {
    char* dir = aicore_relation_model_cache_dir();
    ASSERT_TRUE(dir && std::strstr(dir, "relation_models"));
    aicore_relation_free_buffer(dir);
    aicore_relation_free(nullptr);
    aicore_relation_result_free(nullptr);
    aicore_relation_embeddings_free(nullptr);
    aicore_relation_shutdown();
    aicore_relation_shutdown();
}
