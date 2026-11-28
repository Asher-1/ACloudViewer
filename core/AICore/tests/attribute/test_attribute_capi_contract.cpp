// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <type_traits>

#include "aicore/attribute_capi.h"

TEST(AttributeCApiContract, StandardLayoutAndLocalPresetAbi) {
    static_assert(std::is_standard_layout<aicore_attribute_options>::value,
                  "C ABI");
    static_assert(std::is_standard_layout<aicore_attribute_result>::value,
                  "C ABI");
    static_assert(std::is_standard_layout<aicore_attribute_preset_entry>::value,
                  "C ABI");
    EXPECT_EQ(aicore_attribute_preset_count(), 2)
            << "Missing local person/vehicle recipes";
    for (int i = 0; i < aicore_attribute_preset_count(); ++i) {
        const auto* preset = aicore_attribute_preset_at(i);
        EXPECT_TRUE(preset && preset->struct_size == sizeof(*preset) &&
                    preset->abi_version == 1)
                << "Invalid local recipe ABI";
        EXPECT_TRUE(preset && std::strstr(preset->filename, "-f32.gguf") &&
                    std::strlen(preset->source_sha256) == 64)
                << "Recipes must not represent unvalidated quantized download "
                   "assets";
    }
    EXPECT_TRUE(!aicore_attribute_preset_at(-1) &&
                !aicore_attribute_preset_at(2))
            << "Recipe index out of bounds";
}

TEST(AttributeCApiContract, OwnedModelCacheDirectory) {
    char* directory = aicore_attribute_model_cache_dir();
    EXPECT_TRUE(directory && std::strstr(directory, "attribute_models"))
            << "Missing owning model cache directory";
    aicore_attribute_free_buffer(directory);
    aicore_attribute_free_buffer(nullptr);
}

TEST(AttributeCApiContract, DefaultOptionsDoNotBindModel) {
    const auto defaults = aicore_attribute_default_options();
    EXPECT_TRUE(defaults.struct_size == sizeof(defaults) &&
                defaults.abi_version == AICORE_ATTRIBUTE_ABI_VERSION)
            << "Invalid default options ABI";
    EXPECT_EQ(defaults.model_path, nullptr)
            << "Defaults must not bind a developer-machine model";
}

TEST(AttributeCApiContract, InvalidOptionsAndMissingModelsExposeErrors) {
    const auto defaults = aicore_attribute_default_options();
    for (int kind = 0; kind < 4; ++kind) {
        auto options = defaults;
        options.device = "cpu";
        if (kind == 0) options.struct_size = 0;
        if (kind == 1) options.abi_version = 0;
        if (kind == 2) options.abi_version = AICORE_ATTRIBUTE_ABI_VERSION + 1;
        if (kind == 3) options.model_path = "attribute-contract-missing.gguf";
        auto* context = aicore_attribute_create(&options);
        EXPECT_EQ(aicore_attribute_ready(context), 0)
                << "Invalid arguments or missing models must not be ready";
        EXPECT_TRUE(aicore_attribute_last_error(context) &&
                    std::strlen(aicore_attribute_last_error(context)) > 0)
                << "Failures must expose an error";
        EXPECT_EQ(std::strlen(aicore_attribute_context_device(context)), 0)
                << "Failed loading must not report a device";
        aicore_attribute_result* output = nullptr;
        EXPECT_TRUE(aicore_attribute_run(context, nullptr, &output) == -1 &&
                    !output)
                << "Invalid runs must clear the output";
        aicore_pipeline_timings timing{};
        timing.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
        EXPECT_EQ(aicore_attribute_last_pipeline_timings(context, &timing), -1)
                << "No timing may be reported before inference";
        aicore_attribute_free(context);
    }
}

TEST(AttributeCApiContract, NullRunQueriesAndRelease) {
    aicore_attribute_result* output = nullptr;
    EXPECT_TRUE(aicore_attribute_run(nullptr, nullptr, &output) == -1 &&
                !output)
            << "NULL run did not fail";
    EXPECT_EQ(aicore_attribute_ready(nullptr), 0) << "NULL ready";
    EXPECT_NE(aicore_attribute_last_error(nullptr), nullptr)
            << "NULL last_error";
    EXPECT_EQ(aicore_attribute_last_pipeline_timings(nullptr, nullptr), -1)
            << "NULL timing";
    aicore_attribute_result_free(nullptr);
    aicore_attribute_free(nullptr);
}

TEST(AttributeCApiContract, Shutdown) { aicore_attribute_shutdown(); }
