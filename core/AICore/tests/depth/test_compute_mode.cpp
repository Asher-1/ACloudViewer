// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "tasks/depth/compute_mode.hpp"

class DepthComputeMode : public ::testing::Test {
protected:
    void TearDown() override { aicore::depth::set_gpu_mode(false); }
};

TEST_F(DepthComputeMode, DefaultsToCpu) {
    EXPECT_FALSE(aicore::depth::gpu_mode());
}

TEST_F(DepthComputeMode, CanEnableAndClearGpuMode) {
    aicore::depth::set_gpu_mode(true);
    EXPECT_TRUE(aicore::depth::gpu_mode());
    aicore::depth::set_gpu_mode(false);
    EXPECT_FALSE(aicore::depth::gpu_mode());
}
