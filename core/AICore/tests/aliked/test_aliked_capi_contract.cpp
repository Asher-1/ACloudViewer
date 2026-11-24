// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Contract tests for aicore_aliked_* (no GGUF required).

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>

#include "aicore/aliked_capi.h"
#include "aicore/lightglue_capi.h"

TEST(AlikedCApiContract, AbiAndNullQuantization) {
    EXPECT_GE(aicore_aliked_abi_version(), 2);
    EXPECT_NE(aicore_aliked_quantize_gguf(nullptr, nullptr, nullptr), 0);
}

TEST(AlikedCApiContract, NullReleaseLoadAndQueries) {
    aicore_aliked_free(nullptr);
    aicore_aliked_options_free(nullptr);
    aicore_aliked_free_buffer(nullptr);

    EXPECT_EQ(aicore_aliked_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_aliked_is_ready(nullptr), 0);
    EXPECT_NE(aicore_aliked_info_json(nullptr), nullptr);
}

TEST(AlikedCApiContract, OptionsLifecycle) {
    aicore_aliked_options* opts = aicore_aliked_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_aliked_options_set_device(opts, "cpu");
    aicore_aliked_options_set_threads(opts, 1);
    aicore_aliked_options_set_max_keypoints(opts, 512);
    aicore_aliked_options_set_resize_long_edge(opts, 1024);
    aicore_aliked_options_free(opts);
}

TEST(AlikedCApiContract, NullFeatureExtractionFails) {
    aicore_lightglue_features features{};
    EXPECT_NE(aicore_aliked_extract_rgb(nullptr, nullptr, 0, 0, 0, &features),
              0);
    /* F-01 batch A: image_view entry is NULL-safe. */
    EXPECT_NE(aicore_aliked_extract_image_view(nullptr, nullptr, &features), 0);
}

TEST(AlikedCApiContract, OwnedModelCacheDirectory) {
    char* dir = aicore_aliked_model_cache_dir();
    EXPECT_NE(dir, nullptr);
    EXPECT_GT(std::strlen(dir), 0);
    aicore_aliked_free_buffer(dir);
}
