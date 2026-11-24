// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstring>
#include <set>
#include <string>

#include "aicore/runtime_capi.h"
#include "aicore/tagging_capi.h"
// Frozen declaration from the original public header, independently retained:
// changing a by-value return size breaks callers before infer can validate it.
struct OriginalTaggingRequest {
    uint32_t struct_size;
    uint32_t abi_version;
    float threshold;
    const char* specified_tags;
    float threshold_epsilon;
};
static_assert(sizeof(aicore_tagging_request) == sizeof(OriginalTaggingRequest),
              "original default_request return ABI changed");
static_assert(alignof(aicore_tagging_request) ==
                      alignof(OriginalTaggingRequest),
              "original request alignment changed");

#include <gtest/gtest.h>

TEST(TaggingCApiContract, NullReleaseAndReady) {
    aicore_tagging_free(nullptr);
    aicore_tagging_result_free(nullptr);
    aicore_tagging_free_buffer(nullptr);
    EXPECT_FALSE(aicore_tagging_ready(nullptr));
}

TEST(TaggingCApiContract, CatalogInventoryAndCaptionSupport) {
    EXPECT_EQ(aicore_tagging_model_count(), 9);
    EXPECT_EQ(aicore_tagging_model_at(-1), nullptr);
    EXPECT_EQ(aicore_tagging_model_at(9), nullptr);
    std::set<std::string> names;
    int captions = 0;
    for (int i = 0; i < 9; ++i) {
        const auto* e = aicore_tagging_model_at(i);
        EXPECT_TRUE(e && e->size_bytes > 0);
        if (!e) continue;
        EXPECT_EQ(std::strlen(e->sha256), 64);
        EXPECT_TRUE(
                std::strstr(e->download_url,
                            "huggingface.co/Asher-1/RAM_GGUF/resolve/main/"));
        names.insert(e->filename);
        captions += e->supports_caption;
    }
    EXPECT_TRUE(names.size() == 9 && captions == 3);
}

TEST(TaggingCApiContract, OriginalAndExtendedRequestDefaults) {
    auto opts = aicore_tagging_default_options();
    auto request = aicore_tagging_default_request();
    EXPECT_TRUE(opts.struct_size == sizeof(opts) && request.threshold == -1);
    EXPECT_EQ(opts.cpu_precision, AICORE_TAGGING_CPU_AUTO);
    const auto extended = aicore_tagging_default_request_v2();
    EXPECT_EQ(extended.resize_mode, AICORE_TAGGING_RESIZE_OFFICIAL);
    EXPECT_EQ(extended.struct_size, sizeof(extended));
    EXPECT_GE(offsetof(aicore_tagging_request_v2, resize_mode),
              sizeof(request));
}

TEST(TaggingCApiContract,
     InvalidOptionsExposeErrorsAndClearBothRequestOutputs) {
    auto opts = aicore_tagging_default_options();
    auto request = aicore_tagging_default_request();
    const auto extended = aicore_tagging_default_request_v2();
    for (int mode = 0; mode < 5; ++mode) {
        auto invalid = opts;
        if (mode == 1) invalid.struct_size = 0;
        if (mode == 2) invalid.abi_version = 999;
        if (mode == 3) invalid.max_length = 0;
        if (mode == 4) invalid.cpu_precision = 99;
        auto* ctx = aicore_tagging_create(mode == 0 ? nullptr : &invalid);
        EXPECT_TRUE(ctx && !aicore_tagging_ready(ctx));
        EXPECT_GT(std::strlen(aicore_tagging_last_error(ctx)), 0);
        auto* result = reinterpret_cast<aicore_tagging_result*>(1);
        EXPECT_TRUE(aicore_tagging_infer_image(ctx, nullptr, &request,
                                               &result) == -1 &&
                    result == nullptr);
        result = reinterpret_cast<aicore_tagging_result*>(1);
        EXPECT_TRUE(aicore_tagging_infer_image_v2(ctx, nullptr, &extended,
                                                  &result) == -1 &&
                    !result);
        aicore_pipeline_timings timings{};
        EXPECT_EQ(aicore_tagging_last_pipeline_timings(ctx, &timings), -1);
        aicore_tagging_free(ctx);
    }
}

TEST(TaggingCApiContract, OwnedCacheAndIdempotentShutdown) {
    char* dir = aicore_tagging_model_cache_dir();
    EXPECT_TRUE(dir && std::strstr(dir, "ram_models"));
    aicore_tagging_free_buffer(dir);
    aicore_tagging_shutdown();
    aicore_tagging_shutdown();
}
