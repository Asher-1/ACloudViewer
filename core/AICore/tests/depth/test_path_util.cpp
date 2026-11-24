// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "common/model_cache.hpp"

TEST(DepthModelCache, UsesTaskSpecificDirectory) {
    const std::string dir = aicore::depth_model_cache_dir();
    EXPECT_FALSE(dir.empty());
    EXPECT_NE(dir.find("da3_models"), std::string::npos) << dir;
}
