// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Part A gate: DualDPT AUXILIARY ray head against the dumped reference.
// Feeds the dumped feat_5/7/9/11 (same fixed 224 fixture as test_dpt_head) into
// the independent aux pyramid, then bit-exact-parity-gates ray (6ch, identity)
// + ray_conf (1ch, expp1) vs dumps/reference_rays.gguf. Uses the opt-in
// --with-aux GGUF.
#include <gtest/gtest.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "tasks/depth/backend.hpp"
#include "tasks/depth/dpt_head.hpp"
#include "tasks/depth/model_loader.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, HeadRays) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF_AUX");
    const char* base = std::getenv(
            "AICORE_TEST_DEPTH_BASELINE");  // feats (reference.gguf)
    const char* rays = std::getenv(
            "AICORE_TEST_DEPTH_BASELINE_RAYS");  // gold ray/ray_conf
    if (!gguf || !base || !rays)
        GTEST_SKIP() << "Missing configured model/reference assets";
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf))
        FAIL() << "Required operation or numeric contract failed";
    aicore::depth::Backend be;

    const int H = 224, W = 224;
    const int Ls[4] = {5, 7, 9, 11};
    std::vector<std::vector<float>> feats(4);
    std::vector<int64_t> s;
    for (int i = 0; i < 4; ++i)
        if (!da_parity::load_baseline(
                    base, std::string("feat_") + std::to_string(Ls[i]),
                    feats[i], s))
            FAIL() << "Required operation or numeric contract failed";

    aicore::depth::DptHead head(ml, be);
    std::vector<float> ray, ray_conf;
    int ray_h = 0, ray_w = 0;
    if (!head.rays(feats, H, W, ray, ray_conf, ray_h, ray_w)) {
        std::fprintf(stderr, "rays() failed (aux tensors present? %s)\n", gguf);
        FAIL() << "Required operation or numeric contract failed";
    }
    std::fprintf(stderr,
                 "[rays] aux resolution H=%d W=%d ray=%zu ray_conf=%zu\n",
                 ray_h, ray_w, ray.size(), ray_conf.size());

    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(rays, "ray", ref, s))
            FAIL() << "Required operation or numeric contract failed";
        EXPECT_TRUE(da_parity::compare(ray, ref, "ray", 2e-3f, 2e-3f));
    }
    {
        std::vector<float> ref;
        if (!da_parity::load_baseline(rays, "ray_conf", ref, s))
            FAIL() << "Required operation or numeric contract failed";
        EXPECT_TRUE(
                da_parity::compare(ray_conf, ref, "ray_conf", 2e-3f, 2e-3f));
    }
}
