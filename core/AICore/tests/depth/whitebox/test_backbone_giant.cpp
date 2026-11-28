// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M5 gate: GIANT (vitg, 40 layers, embed 1536, SwiGLU FFN) backbone parity.
// Runs the metadata-driven DinoBackbone::forward (S=1) on the giant GGUF and
// compares feat_g_{19,27,33,39} + cam_g_{19,27,33,39} against the dump.
// SLOW: 40-layer CPU forward takes tens of seconds. SKIP (77) if artifacts
// absent.
#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include "tasks/depth/dino_backbone.hpp"
#include "tasks/depth/model_loader.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, BackboneGiant) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF_GIANT");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_GIANT");
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
    const int Ls[4] = {19, 27, 33, 39};
    for (int i = 0; i < 4; ++i) {
        std::vector<float> rf, rc;
        ASSERT_TRUE(da_parity::load_baseline(
                base, std::string("feat_g_") + std::to_string(Ls[i]), rf, s));
        ASSERT_TRUE(da_parity::load_baseline(
                base, std::string("cam_g_") + std::to_string(Ls[i]), rc, s));
        EXPECT_TRUE(da_parity::compare(
                feats[i], rf,
                (std::string("feat_g_") + std::to_string(Ls[i])).c_str(), 2e-3f,
                2e-3f));
        EXPECT_TRUE(da_parity::compare(
                cams[i], rc,
                (std::string("cam_g_") + std::to_string(Ls[i])).c_str(), 2e-3f,
                2e-3f));
    }
}
