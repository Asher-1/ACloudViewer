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

#include "aicore/deeplsd_capi.h"

TEST(DeeplsdCApiContract, AbiAndNullReleaseLoadQueries) {
    EXPECT_GE(aicore_deeplsd_abi_version(), 1);

    aicore_deeplsd_free(nullptr);
    aicore_deeplsd_options_free(nullptr);
    aicore_deeplsd_free_buffer(nullptr);

    EXPECT_EQ(aicore_deeplsd_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_deeplsd_is_ready(nullptr), 0);
    EXPECT_NE(aicore_deeplsd_info_json(nullptr), nullptr);
}

TEST(DeeplsdCApiContract, OptionsLifecycle) {
    aicore_deeplsd_options* opts = aicore_deeplsd_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_deeplsd_options_set_device(opts, "cpu");
    aicore_deeplsd_options_set_threads(opts, 1);
    aicore_deeplsd_options_free(opts);
}

TEST(DeeplsdCApiContract, NullGrayAndImageViewExtractionFails) {
    float* df = nullptr;
    float* ang = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    EXPECT_NE(aicore_deeplsd_extract_gray(nullptr, nullptr, 0, 0, 0, &df, &ang,
                                          &w, &h),
              0);
    /* F-01 batch A: image_view entry is NULL-safe. */
    EXPECT_NE(aicore_deeplsd_extract_image_view(nullptr, nullptr, &df, &ang, &w,
                                                &h),
              0);
}

TEST(DeeplsdCApiContract, NullQuantizationAndCpuWarmup) {
    EXPECT_NE(aicore_deeplsd_quantize(nullptr, nullptr, nullptr), 0);
    EXPECT_EQ(aicore_deeplsd_warmup_backend("cpu"), 0);
}

TEST(DeeplsdCApiContract, OwnedModelCacheDirectory) {
    char* dir = aicore_deeplsd_model_cache_dir();
    EXPECT_TRUE(dir != nullptr && std::strlen(dir) > 0);
    aicore_deeplsd_free_buffer(dir);
}
