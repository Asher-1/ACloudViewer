// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <vector>

#include "tasks/depth/cam_pose.hpp"
#include "tasks/depth/model_loader.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, CamPose) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base)
        GTEST_SKIP() << "Required model/reference environment is unset";
    aicore::depth::ModelLoader ml;
    ASSERT_TRUE(ml.load(gguf));
    std::vector<float> ct;
    std::vector<int64_t> s;
    ASSERT_TRUE(da_parity::load_baseline(base, "cam_token_11", ct, s));
    aicore::depth::Backend be;
    aicore::depth::CamPose cp(ml, be);
    std::array<float, 9> pe;
    std::array<float, 12> ext;
    std::array<float, 9> K;
    ASSERT_TRUE(cp.pose(ct, 224, 224, pe, ext, K));
    std::vector<float> rpe, rext, rK;
    ASSERT_TRUE(da_parity::load_baseline(base, "pose_enc", rpe, s));
    ASSERT_TRUE(da_parity::load_baseline(base, "extrinsics", rext, s));
    ASSERT_TRUE(da_parity::load_baseline(base, "intrinsics", rK, s));
    std::vector<float> vpe(pe.begin(), pe.end()), vext(ext.begin(), ext.end()),
            vK(K.begin(), K.end());
    EXPECT_TRUE(da_parity::compare(vpe, rpe, "pose_enc", 2e-3f, 2e-3f));
    EXPECT_TRUE(da_parity::compare(vext, rext, "extrinsics", 2e-3f, 2e-3f));
    EXPECT_TRUE(da_parity::compare(vK, rK, "intrinsics", 2e-3f, 2e-3f));
}
