// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "aicore/aicore.h"

TEST(AicoreImageDepth, DefaultDeviceIsAvailable) {
    EXPECT_TRUE(aicore::depth::ImageDepth::isAvailable());
}

TEST(AicoreUmbrellaHeader, ExposesSupportedDepthAbi) {
    EXPECT_GE(aicore_depth_abi_version(), 5);
}

TEST(AicoreUmbrellaHeader, ExposesSupportedGaussianAbi) {
    EXPECT_GE(aicore_gaussian_abi_version(), 1);
}
