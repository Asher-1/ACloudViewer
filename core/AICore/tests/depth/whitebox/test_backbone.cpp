// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "tasks/depth/dino_backbone.hpp"
#include "tasks/depth/model_loader.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, Backbone) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base)
        GTEST_SKIP() << "Required model/reference environment is unset";
    aicore::depth::ModelLoader ml;
    ASSERT_TRUE(ml.load(gguf));
    std::vector<float> img;
    std::vector<int64_t> s;
    ASSERT_TRUE(da_parity::load_baseline(base, "input_image", img, s));
    int HW = (int)(img.size() / 3);
    int H = (int)std::lround(std::sqrt((double)HW)), W = H;
    aicore::depth::Backend be;
    aicore::depth::DinoBackbone bb(ml, be);
    std::vector<std::vector<float>> feats, cams;
    ASSERT_TRUE(bb.forward(img, H, W, feats, cams));
    ASSERT_EQ(feats.size(), 4u);
    ASSERT_EQ(cams.size(), 4u);
    const int Ls[4] = {5, 7, 9, 11};
    for (int i = 0; i < 4; ++i) {
        std::vector<float> rf, rc;
        ASSERT_TRUE(da_parity::load_baseline(
                base, std::string("feat_") + std::to_string(Ls[i]), rf, s));
        ASSERT_TRUE(da_parity::load_baseline(
                base, std::string("cam_token_") + std::to_string(Ls[i]), rc,
                s));
        EXPECT_TRUE(da_parity::compare(
                feats[i], rf,
                (std::string("feat_") + std::to_string(Ls[i])).c_str(), 2e-3f,
                2e-3f));
        EXPECT_TRUE(da_parity::compare(
                cams[i], rc,
                (std::string("cam_") + std::to_string(Ls[i])).c_str(), 2e-3f,
                2e-3f));
    }
}
