// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// AnyCalib public-ABI model-free contract: catalog lookup, allocation release
// and error-output ownership.
#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <string>

#include "aicore/anycalib_capi.h"

TEST(AnycalibCApiContract, AbiAndCatalogRoundTrip) {
    EXPECT_EQ(aicore_anycalib_abi_version(), 1);
    EXPECT_EQ(aicore_anycalib_model_by_filename(nullptr), nullptr);
    EXPECT_EQ(aicore_anycalib_model_by_filename("not-a-model.gguf"), nullptr);
    EXPECT_EQ(aicore_anycalib_model_at(-1), nullptr);
    EXPECT_EQ(aicore_anycalib_model_at(aicore_anycalib_model_count()), nullptr);
    EXPECT_GT(aicore_anycalib_model_count(), 0);
    for (int i = 0; i < aicore_anycalib_model_count(); ++i) {
        const auto* entry = aicore_anycalib_model_at(i);
        EXPECT_TRUE(entry != nullptr && entry->filename != nullptr);
        if (entry && entry->filename)
            EXPECT_EQ(aicore_anycalib_model_by_filename(entry->filename),
                      entry);
    }
}

TEST(AnycalibCApiContract, OwnedModelCacheDirectory) {
    char* cache = aicore_anycalib_model_cache_dir();
    EXPECT_NE(cache, nullptr);
    if (cache) {
        EXPECT_EQ(std::filesystem::path(cache).filename(), "anycalib_models");
        aicore_anycalib_free_buffer(cache);
    }
}

TEST(AnycalibCApiContract, MissingPathErrorContextClearsRunOutputs) {
    aicore_anycalib_options_set_threads(nullptr, 2);
    auto* options = aicore_anycalib_options_new();
    EXPECT_NE(options, nullptr);
    aicore_anycalib_options_set_threads(options, 2);
    aicore_anycalib_options_set_device(options, "cpu");
    auto* ctx = aicore_anycalib_load_opts(nullptr, options);
    aicore_anycalib_options_free(options);
    EXPECT_TRUE(ctx != nullptr && aicore_anycalib_is_ready(ctx) == 0);
    const char* error = aicore_anycalib_last_error(ctx);
    EXPECT_TRUE(error && std::strstr(error, "model path"));
    float* rays = reinterpret_cast<float*>(1);
    float* tangent = reinterpret_cast<float*>(1);
    int32_t count = 99;
    EXPECT_LT(aicore_anycalib_run(ctx, nullptr, &rays, &tangent, &count), 0);
    EXPECT_TRUE(rays == nullptr && tangent == nullptr && count == 0);
    aicore_anycalib_free(ctx);
}

TEST(AnycalibCApiContract, ShutdownIsIdempotent) {
    aicore_anycalib_shutdown();
    aicore_anycalib_shutdown();
}
