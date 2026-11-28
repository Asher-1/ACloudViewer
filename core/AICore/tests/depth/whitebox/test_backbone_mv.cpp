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

TEST(DepthWhitebox, BackboneMv) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE_MV");
    if (!gguf || !base)
        GTEST_SKIP() << "Required model/reference environment is unset";
    aicore::depth::ModelLoader ml;
    ASSERT_TRUE(ml.load(gguf));

    // ImageNet normalization (matches
    // scripts/da3_reference.fixed_input_multiview).
    const float mean[3] = {0.485f, 0.456f, 0.406f},
                std_[3] = {0.229f, 0.224f, 0.225f};
    const int H = 224, W = 224, S = 2;
    std::vector<std::vector<float>> views(S);
    for (int v = 0; v < S; ++v) {
        std::vector<float> raw;
        std::vector<int64_t> s;  // raw_mv_v stored HWC [224,224,3] (0..255)
        ASSERT_TRUE(da_parity::load_baseline(
                base, std::string("raw_mv_") + std::to_string(v), raw, s));
        ASSERT_EQ(raw.size(), static_cast<size_t>(3 * H * W));
        std::vector<float> chw((size_t)3 * H * W);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                for (int c = 0; c < 3; ++c) {
                    float px = raw[((size_t)y * W + x) * 3 + c] / 255.0f;
                    chw[((size_t)c * H + y) * W + x] = (px - mean[c]) / std_[c];
                }
        views[v] = std::move(chw);
    }

    aicore::depth::Backend be;
    aicore::depth::DinoBackbone bb(ml, be);
    std::vector<std::vector<std::vector<float>>> feats, cams;
    ASSERT_TRUE(bb.forward_mv(views, H, W, feats, cams));

    ASSERT_EQ(feats.size(), 4u);
    ASSERT_EQ(cams.size(), 4u);
    const int Ls[4] = {5, 7, 9, 11};
    const size_t feat_view = 256 * 1536, cam_view = 1536;
    for (int i = 0; i < 4; ++i) {
        std::vector<float> rf, rc;
        std::vector<int64_t> s;
        ASSERT_TRUE(da_parity::load_baseline(
                base, std::string("feat_mv_") + std::to_string(Ls[i]), rf, s));
        ASSERT_TRUE(da_parity::load_baseline(
                base, std::string("cam_mv_") + std::to_string(Ls[i]), rc, s));
        ASSERT_EQ(rf.size(), static_cast<size_t>(S) * feat_view);
        ASSERT_EQ(rc.size(), static_cast<size_t>(S) * cam_view);
        ASSERT_EQ(feats[i].size(), static_cast<size_t>(S));
        ASSERT_EQ(cams[i].size(), static_cast<size_t>(S));
        for (int v = 0; v < S; ++v) {
            std::vector<float> rf_v(rf.begin() + (size_t)v * feat_view,
                                    rf.begin() + (size_t)(v + 1) * feat_view);
            std::vector<float> rc_v(rc.begin() + (size_t)v * cam_view,
                                    rc.begin() + (size_t)(v + 1) * cam_view);
            EXPECT_TRUE(da_parity::compare(
                    feats[i][v], rf_v,
                    (std::string("feat_mv_") + std::to_string(Ls[i]) + "_v" +
                     std::to_string(v))
                            .c_str(),
                    2e-3f, 2e-3f));
            EXPECT_TRUE(da_parity::compare(
                    cams[i][v], rc_v,
                    (std::string("cam_mv_") + std::to_string(Ls[i]) + "_v" +
                     std::to_string(v))
                            .c_str(),
                    2e-3f, 2e-3f));
        }
    }
}
