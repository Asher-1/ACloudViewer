// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "tasks/depth/uv_posembed.hpp"
#include "tests/depth/whitebox/parity.hpp"

TEST(DepthWhitebox, UvPosembed) {
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!base) GTEST_SKIP() << "Required model/reference environment is unset";
    std::vector<float> ref;
    std::vector<int64_t> s;
    ASSERT_TRUE(da_parity::load_baseline(base, "uv_embed_64", ref, s));
    std::vector<float> got =
            aicore::depth::uv_pos_embed(/*pw=*/224, /*ph=*/224, /*C=*/64,
                                        /*aspect=*/1.0f, 100.f);
    EXPECT_TRUE(da_parity::compare(got, ref, "uv_embed_64", 1e-5f, 1e-5f));
}
