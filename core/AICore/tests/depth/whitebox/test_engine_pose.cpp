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

#include "tasks/depth/engine.hpp"
#include "tasks/depth/image_io.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, EnginePose) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base)
        GTEST_SKIP() << "Missing configured model/reference assets";
    std::vector<float> raw, rext, rintr;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "raw_image", raw, s))
        FAIL() << "Configured reference asset could not be loaded";
    da_parity::load_baseline(base, "extrinsics", rext, s);
    da_parity::load_baseline(base, "intrinsics", rintr, s);
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
    std::array<float, 12> ext;
    std::array<float, 9> intr;
    int H, W;
    if (!eng->depth_pose(img, depth, conf, ext, intr, H, W))
        FAIL() << "Required operation or numeric contract failed";
    std::vector<float> vext(ext.begin(), ext.end()),
            vintr(intr.begin(), intr.end());
    EXPECT_TRUE(
            da_parity::compare(vext, rext, "engine_extrinsics", 5e-3f, 5e-3f));
    EXPECT_TRUE(da_parity::compare(vintr, rintr, "engine_intrinsics", 5e-3f,
                                   5e-3f));
}
