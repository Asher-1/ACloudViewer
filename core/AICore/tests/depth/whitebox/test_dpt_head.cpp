// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// M2 component gate: full DualDPT MAIN depth path against the dumped reference.
// Feeds the dumped feat_5/7/9/11 as the head input (isolates head from
// backbone), then gates the 4 post-resize stages, the fused map, and the final
// depth/conf.
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "tasks/depth/backend.hpp"
#include "tasks/depth/dpt_head.hpp"
#include "tasks/depth/model_loader.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, DptHead) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base)
        GTEST_SKIP() << "Missing configured model/reference assets";
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf))
        FAIL() << "Required operation or numeric contract failed";
    aicore::depth::Backend be;

    const int H = 224, W = 224;
    const int Ls[4] = {5, 7, 9, 11};
    std::vector<std::vector<float>> feats(4);
    std::vector<int64_t> s;
    for (int i = 0; i < 4; ++i) {
        if (!da_parity::load_baseline(
                    base, std::string("feat_") + std::to_string(Ls[i]),
                    feats[i], s))
            FAIL() << "Required operation or numeric contract failed";
    }

    aicore::depth::DptHead head(ml, be);
    std::vector<float> depth, conf, fused;
    std::vector<std::vector<float>> stages;
    if (!head.depth_debug(feats, H, W, depth, conf, stages, fused)) {
        std::fprintf(stderr, "depth_debug failed\n");
        FAIL() << "Required operation or numeric contract failed";
    }

    ASSERT_EQ(stages.size(), 4u);
    // ISOLATION: post-resize stages.
    for (int i = 0; i < 4; ++i) {
        std::vector<float> ref;
        if (!da_parity::load_baseline(
                    base, std::string("head_stage") + std::to_string(i), ref,
                    s))
            FAIL() << "Required operation or numeric contract failed";
        EXPECT_TRUE(da_parity::compare(
                stages[i], ref,
                (std::string("head_stage") + std::to_string(i)).c_str(), 2e-3f,
                2e-3f));
    }
    // ISOLATION: fused (post output_conv1).
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "head_fused", ref, s))
            FAIL() << "Required operation or numeric contract failed";
        EXPECT_TRUE(da_parity::compare(fused, ref, "head_fused", 2e-3f, 2e-3f));
    }
    // MAIN gate: depth + conf.
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "head_depth", ref, s))
            FAIL() << "Required operation or numeric contract failed";
        EXPECT_TRUE(da_parity::compare(depth, ref, "head_depth", 2e-3f, 2e-3f));
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(base, "head_depth_conf", ref, s))
            FAIL() << "Required operation or numeric contract failed";
        EXPECT_TRUE(
                da_parity::compare(conf, ref, "head_depth_conf", 2e-3f, 2e-3f));
    }
}
