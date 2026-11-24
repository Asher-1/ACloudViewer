// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstring>
#include <set>
#include <string>

#include "aicore/model_catalog_capi.h"

namespace {
class SharedModelCatalog : public testing::TestWithParam<aicore_model_family> {
};

TEST_P(SharedModelCatalog, PublishedRowsHaveUniquePinnedHttpsAssets) {
    const auto family = GetParam();
    const int count = aicore_model_count(family);
    ASSERT_GT(count, 0);
    std::set<std::string> filenames;
    for (int i = 0; i < count; ++i) {
        SCOPED_TRACE(i);
        const auto* entry = aicore_model_at(family, i);
        ASSERT_NE(entry, nullptr);
        ASSERT_NE(entry->filename, nullptr);
        EXPECT_NE(*entry->filename, '\0');
        ASSERT_NE(entry->display_name, nullptr);
        EXPECT_NE(*entry->display_name, '\0');
        ASSERT_NE(entry->download_url, nullptr);
        EXPECT_EQ(std::strncmp(entry->download_url, "https://", 8), 0);
        ASSERT_NE(entry->sha256, nullptr);
        EXPECT_EQ(std::strlen(entry->sha256), 64u);
        EXPECT_TRUE(filenames.insert(entry->filename).second);
    }
}

TEST_P(SharedModelCatalog, RejectsInvalidIndicesAndLooksUpFirstFilename) {
    const auto family = GetParam();
    const int count = aicore_model_count(family);
    ASSERT_GT(count, 0);
    EXPECT_EQ(aicore_model_at(family, -1), nullptr);
    EXPECT_EQ(aicore_model_at(family, count), nullptr);
    const auto* first = aicore_model_at(family, 0);
    ASSERT_NE(first, nullptr);
    ASSERT_NE(first->filename, nullptr);
    const std::string filename = first->filename;
    const auto* found = aicore_model_by_filename(family, filename.c_str());
    ASSERT_NE(found, nullptr);
    ASSERT_NE(found->filename, nullptr);
    EXPECT_EQ(filename, found->filename);
}

INSTANTIATE_TEST_SUITE_P(Families,
                         SharedModelCatalog,
                         testing::Values(AICORE_MODEL_FAMILY_DEPTH,
                                         AICORE_MODEL_FAMILY_DEEPLSD,
                                         AICORE_MODEL_FAMILY_LIGHTGLUE,
                                         AICORE_MODEL_FAMILY_GAUSSIAN,
                                         AICORE_MODEL_FAMILY_ALIKED));

TEST(SharedModelCatalogLookup, RejectsUnknownFilename) {
    EXPECT_EQ(
            aicore_model_by_filename(AICORE_MODEL_FAMILY_DEPTH, "missing.gguf"),
            nullptr);
}
}  // namespace
