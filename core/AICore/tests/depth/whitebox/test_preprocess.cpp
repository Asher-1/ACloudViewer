// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "tasks/depth/model_loader.hpp"
#include "tasks/depth/preprocess.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, Preprocess) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!gguf || !base)
        GTEST_SKIP() << "Required model/reference environment is unset";
    aicore::depth::ModelLoader ml;
    ASSERT_TRUE(ml.load(gguf));
    std::vector<float> raw, ref;
    std::vector<int64_t> s;
    ASSERT_TRUE(da_parity::load_baseline(base, "raw_image", raw,
                                         s));  // needs regenerated dump
    ASSERT_TRUE(da_parity::load_baseline(base, "input_image", ref, s));
    // raw is (224,224,3) HWC floats 0..255
    const int H = 224, W = 224;
    aicore::depth::Image img;
    img.w = W;
    img.h = H;
    img.rgb.resize((size_t)W * H * 3);
    ASSERT_EQ(raw.size(), img.rgb.size());
    for (size_t i = 0; i < img.rgb.size(); ++i)
        img.rgb[i] = (unsigned char)(raw[i] + 0.5f);
    aicore::depth::Preprocessed p;
    ASSERT_TRUE(aicore::depth::preprocess(img, ml.config(), p));
    if (p.H != H || p.W != W) {
        std::fprintf(stderr, "wrong dims %dx%d\n", p.H, p.W);
        FAIL() << "Model operation failed";
    }
    EXPECT_TRUE(da_parity::compare(p.chw, ref, "preprocess", 1e-3f, 1e-3f));
}
