// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Model-free contract of the public table C ABI: catalog + digest closure,
// NULL safety of every accessor, option validation and shutdown idempotence.

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "aicore/asset_digests.h"
#include "aicore/table_capi.h"

TEST(TableCApiContract, CatalogMatchesPublishedNamesRolesAndDigests) {
    ASSERT_EQ(aicore_table_model_count(), 3);
    const std::string base = aicore_table_model_download_base();
    ASSERT_FALSE(base.empty());
    EXPECT_EQ(base.back(), '/');
    const char* quants[] = {"f32", "f16", "q8_0"};
    for (int i = 0; i < aicore_table_model_count(); ++i) {
        const aicore_table_model_entry* entry = aicore_table_model_at(i);
        ASSERT_NE(entry, nullptr);
        ASSERT_NE(entry->filename, nullptr);
        const std::string name = entry->filename;
        EXPECT_EQ(name,
                  std::string("slanet-plus-table-") + quants[i] + ".gguf");
        ASSERT_NE(entry->download_url, nullptr);
        EXPECT_EQ(std::string(entry->download_url), base + name);
        ASSERT_NE(entry->role, nullptr);
        EXPECT_STREQ(entry->role, "table");
        ASSERT_NE(entry->arch, nullptr);
        EXPECT_STREQ(entry->arch, "slanet_plus");
        EXPECT_TRUE(entry->display_name && entry->display_name[0]);
        const char* sha = aicore::AssetDigestForFile(entry->filename);
        ASSERT_NE(sha, nullptr);
        EXPECT_EQ(std::strlen(sha), 64);
        EXPECT_NE(aicore_table_model_by_filename(entry->filename), nullptr);
    }
    EXPECT_EQ(aicore_table_model_at(-1), nullptr);
    EXPECT_EQ(aicore_table_model_at(3), nullptr);
    EXPECT_EQ(aicore_table_model_by_filename("missing.gguf"), nullptr);
    EXPECT_EQ(aicore_table_model_by_filename(nullptr), nullptr);

    char* cache = aicore_table_model_cache_dir();
    ASSERT_NE(cache, nullptr);
    EXPECT_NE(std::strstr(cache, "ocr_models"), nullptr);
    aicore_table_free_buffer(cache);
    aicore_table_free_buffer(nullptr);
}

TEST(TableCApiContract, NullContextAccessorsReturnEmptyDefaults) {
    EXPECT_EQ(aicore_table_context_create(nullptr), nullptr);
    ASSERT_NE(aicore_table_last_error(), nullptr);
    EXPECT_NE(aicore_table_last_error()[0], '\0');
    aicore_table_context_destroy(nullptr);
    EXPECT_EQ(aicore_table_context_ready(nullptr), 0);
    EXPECT_NE(aicore_table_context_device(nullptr), nullptr);
    EXPECT_EQ(aicore_table_run(nullptr, nullptr, nullptr, nullptr, nullptr),
              nullptr);
    EXPECT_EQ(aicore_table_count(nullptr), 0);
    EXPECT_EQ(aicore_table_at(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_table_cell_count(nullptr), 0);
    EXPECT_EQ(aicore_table_cell_at(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_table_token_at(nullptr, 0, 0), nullptr);
    aicore_table_result_release(nullptr);
    const aicore_pipeline_timings timings = aicore_table_timings(nullptr);
    EXPECT_EQ(timings.valid_fields, 0);
}

TEST(TableCApiContract, MissingModelPathIsRejected) {
    aicore_table_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = AICORE_TABLE_ABI_VERSION;
    options.device = "cpu";
    EXPECT_EQ(aicore_table_context_create(&options), nullptr);
}

TEST(TableCApiContract, MissingModelAndInvalidAbiOptionsAreRejected) {
    aicore_table_options options{};
    options.struct_size = sizeof(options);
    options.abi_version = AICORE_TABLE_ABI_VERSION;
    options.device = "cpu";
    options.model_path = "/nonexistent/slanet-plus-table-f32.gguf";
    EXPECT_EQ(aicore_table_context_create(&options), nullptr);
    ASSERT_NE(aicore_table_last_error(), nullptr);
    EXPECT_GT(std::strlen(aicore_table_last_error()), 0);

    aicore_table_options small = options;
    small.struct_size = sizeof(options) - 1;
    EXPECT_EQ(aicore_table_context_create(&small), nullptr);
    aicore_table_options future_abi = options;
    future_abi.abi_version = AICORE_TABLE_ABI_VERSION + 1;
    EXPECT_EQ(aicore_table_context_create(&future_abi), nullptr);
}

TEST(TableCApiContract, RuntimeShutdownIsIdempotent) {
    aicore_table_runtime_shutdown();
    aicore_table_runtime_shutdown();
}
