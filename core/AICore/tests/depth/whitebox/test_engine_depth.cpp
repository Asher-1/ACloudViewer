// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// E2E fixture gate: dumped raw_image -> full Engine depth -> compare vs
// head_depth.
#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "tasks/depth/engine.hpp"
#include "tasks/depth/image_io.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, EngineDepth) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base)
        GTEST_SKIP() << "Missing configured model/reference assets";
    std::vector<float> raw, ref;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "raw_image", raw, s))
        FAIL() << "Configured reference asset could not be loaded";
    if (!da_parity::load_baseline(base, "head_depth", ref, s))
        FAIL() << "Required operation or numeric contract failed";
    auto eng = aicore::depth::Engine::load(gguf, {});
    if (!eng) FAIL() << "Required operation or numeric contract failed";
    aicore::depth::Image img;
    img.w = 224;
    img.h = 224;
    img.rgb.resize(224 * 224 * 3);
    ASSERT_EQ(raw.size(), img.rgb.size());
    for (size_t i = 0; i < img.rgb.size(); ++i)
        img.rgb[i] = (unsigned char)(raw[i] + 0.5f);
    std::vector<float> depth, conf;
    int H, W;
    if (!eng->depth_image(img, depth, conf, H, W))
        FAIL() << "Required operation or numeric contract failed";
    // full pipeline: preprocess(exact) -> backbone(~4e-4) -> head; accumulated
    // tol
    EXPECT_TRUE(da_parity::compare(depth, ref, "engine_depth", 5e-3f, 5e-3f));
}
