// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Contract tests for aicore_lightglue_* (no GGUF required).

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include "aicore/lightglue_capi.h"

TEST(LightglueCApiContract, AbiAndNullRelease) {
    EXPECT_GE(aicore_lightglue_abi_version(), 2);

    aicore_lightglue_free(nullptr);
    aicore_lightglue_free_matches(nullptr);
    aicore_lightglue_free_buffer(nullptr);
    aicore_lightglue_free_features(nullptr);
    aicore_lightglue_options_free(nullptr);
}

TEST(LightglueCApiContract, NullLoadAndGeometryQueries) {
    EXPECT_EQ(aicore_lightglue_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_lightglue_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_lightglue_info_json(nullptr), nullptr);
    aicore_lightglue_geometry geo{};
    EXPECT_NE(aicore_lightglue_geometry_of(nullptr, &geo), 0);
}

TEST(LightglueCApiContract, OptionsLifecycle) {
    aicore_lightglue_options* opts = aicore_lightglue_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_lightglue_options_set_device(opts, "cpu");
    aicore_lightglue_options_set_threads(opts, 1);
    aicore_lightglue_options_set_min_score(opts, 0.1);
    aicore_lightglue_options_set_matcher_type(opts, 2);
    aicore_lightglue_options_free(opts);
}

TEST(LightglueCApiContract, NullMatchingAndMissingFixtureFail) {
    aicore_lightglue_match* matches = nullptr;
    int32_t n_matches = 0;
    EXPECT_NE(aicore_lightglue_run_match(nullptr, nullptr, nullptr, &matches,
                                         &n_matches),
              0);

    aicore_lightglue_features f0{}, f1{};
    EXPECT_NE(aicore_lightglue_load_fixture("/nonexistent.bin", &f0, &f1), 0);
}

TEST(LightglueCApiContract, NullQuantizationFails) {
    EXPECT_NE(aicore_lightglue_quantize_gguf(nullptr, nullptr, nullptr), 0);
}

TEST(LightglueCApiContract, CpuWarmupAndOwnedCacheDirectory) {
    EXPECT_EQ(aicore_lightglue_warmup_backend("cpu"), 0);

    char* dir = aicore_lightglue_model_cache_dir();
    EXPECT_NE(dir, nullptr);
    EXPECT_GT(std::strlen(dir), 0);
    aicore_lightglue_free_buffer(dir);
}
