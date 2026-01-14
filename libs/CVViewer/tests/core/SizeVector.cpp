// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                            -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "cloudViewer/core/SizeVector.h"

#include "tests/Tests.h"

namespace cloudViewer {
namespace tests {

TEST(DynamicSizeVector, Constructor) {
    core::DynamicSizeVector dsv{std::nullopt, 3};
    EXPECT_FALSE(dsv[0].has_value());
    EXPECT_EQ(dsv[1].value(), 3);
}

TEST(DynamicSizeVector, IsCompatible) {
    EXPECT_TRUE(core::SizeVector({}).IsCompatible({}));
    EXPECT_FALSE(core::SizeVector({}).IsCompatible({std::nullopt}));
    EXPECT_TRUE(core::SizeVector({10, 3}).IsCompatible({std::nullopt, 3}));
    EXPECT_FALSE(core::SizeVector({10, 3}).IsCompatible({std::nullopt, 5}));
}

}  // namespace tests
}  // namespace cloudViewer
