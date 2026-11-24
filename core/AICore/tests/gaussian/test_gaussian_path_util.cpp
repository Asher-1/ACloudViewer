// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "common/model_cache.hpp"

TEST(GaussianModelCache, UsesTaskSpecificDirectory) {
    const std::string dir = aicore::gaussian_model_cache_dir();
    EXPECT_FALSE(dir.empty());
    EXPECT_NE(dir.find("freesplatter_models"), std::string::npos) << dir;
}
