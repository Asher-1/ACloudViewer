// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>

#include "aicore/gkd_capi.h"

// Detect request validation needs a loaded model (the options and
// request structs are validated against session state); the model-backed
// checks — no-prompts rejection, struct-size mismatch, multimodal count
// mismatch, repeated-forward stability and context recreation — live in
// test_gkd_capi_load.cpp.

TEST(GkdCApiContract, AbiAndNullReleaseSetters) {
    EXPECT_GE(aicore_gkd_abi_version(), 1);

    // NULL-safe teardown.
    aicore_gkd_free(nullptr);
    aicore_gkd_options_free(nullptr);
    aicore_gkd_free_buffer(nullptr);

    // NULL-safe setters.
    aicore_gkd_options_set_device(nullptr, "cpu");
    aicore_gkd_options_set_threads(nullptr, 1);
    aicore_gkd_options_set_log_level(nullptr, 3);
    aicore_gkd_options_set_dump_dir(nullptr, nullptr);
}

TEST(GkdCApiContract, NullLoadContextResultsAndTimings) {
    EXPECT_EQ(aicore_gkd_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_gkd_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_gkd_last_error(nullptr), nullptr);
    EXPECT_EQ(aicore_gkd_last_timings(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_gkd_last_pipeline_timings(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_gkd_result_keypoint_count(nullptr), 0);
    EXPECT_EQ(aicore_gkd_result_prompt_at(nullptr, 0), nullptr);
    float roi[4] = {0, 0, 0, 0};
    EXPECT_EQ(aicore_gkd_result_roi(nullptr, roi), -1);
    EXPECT_EQ(aicore_gkd_context_image_size(nullptr), 0);
    // Model name is an empty string (never NULL) without a loaded context.
    EXPECT_EQ(std::strlen(aicore_gkd_context_model_name(nullptr)), 0);
    EXPECT_EQ(std::strlen(aicore_gkd_context_device(nullptr)), 0);
    EXPECT_EQ(aicore_gkd_info_json(nullptr), nullptr);
    EXPECT_NE(aicore_gkd_detect_image(nullptr, nullptr, nullptr), 0);
}

TEST(GkdCApiContract, OptionsLifecycle) {
    aicore_gkd_options* opts = aicore_gkd_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_gkd_options_set_device(opts, "cpu");
    aicore_gkd_options_set_threads(opts, 1);
    aicore_gkd_options_set_log_level(opts, 3);
    aicore_gkd_options_set_dump_dir(opts, nullptr);  // cleared / no-op
    aicore_gkd_options_free(opts);
}

TEST(GkdCApiContract, CpuWarmup) {
    EXPECT_EQ(aicore_gkd_warmup_backend("cpu"), 0);
}

TEST(GkdCApiContract, PinnedCatalogAndRecommendedDefault) {
    // Model catalog invariants: URLs match the pinned repo, every row has
    // the exact published size, exactly one recommended default. The
    // deprecated upstream q4_0 build is intentionally not cataloged.
    const int count = aicore_gkd_model_count();
    EXPECT_EQ(count, 4);
    const char* base = aicore_gkd_model_download_base();
    EXPECT_TRUE(base != nullptr &&
                std::strstr(base, "huggingface.co/Asher-1/GKD_GGUF") !=
                        nullptr);
    int recommended = 0;
    for (int i = 0; i < count; ++i) {
        const aicore_gkd_model_entry* e = aicore_gkd_model_at(i);
        EXPECT_NE(e, nullptr);
        if (!e) continue;
        EXPECT_TRUE(e->filename != nullptr && e->filename[0] != '\0');
        EXPECT_TRUE(e->download_url != nullptr &&
                    std::strncmp(e->download_url, base, std::strlen(base)) ==
                            0);
        EXPECT_NE(std::strstr(e->download_url, e->filename), nullptr);
        EXPECT_GT(e->size_bytes, 0);
        EXPECT_EQ(aicore_gkd_model_by_filename(e->filename), e);
        if (std::strstr(e->quant_note, "(recommended)") != nullptr) {
            ++recommended;
            EXPECT_EQ(aicore_gkd_model_default_index(), i);
        }
    }
    EXPECT_EQ(recommended, 1);
    EXPECT_EQ(aicore_gkd_model_at(-1), nullptr);
    EXPECT_EQ(aicore_gkd_model_at(count), nullptr);
    EXPECT_EQ(aicore_gkd_model_by_filename("missing.gguf"), nullptr);
}

TEST(GkdCApiContract, OwnedModelCacheDirectory) {
    char* dir = aicore_gkd_model_cache_dir();
    EXPECT_TRUE(dir != nullptr && std::strlen(dir) > 0 &&
                std::strstr(dir, "gkd_models") != nullptr);
    aicore_gkd_free_buffer(dir);
}
