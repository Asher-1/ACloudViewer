// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <vector>

#include "tasks/depth/dino_backbone.hpp"
#include "tasks/depth/model_loader.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, BackbonePrepare) {
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
    std::vector<float> got;
    ASSERT_TRUE(bb.prepare_tokens(img, H, W, got));
    std::vector<float> ref;
    ASSERT_TRUE(da_parity::load_baseline(base, "pos_embed_added", ref, s));
    EXPECT_TRUE(da_parity::compare(got, ref, "pos_embed_added", 2e-3f, 2e-3f));
}
