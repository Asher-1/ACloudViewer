// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "common/ggml_extend.hpp"
#include "tasks/depth/backend.hpp"

TEST(DepthGgmlExtend, LayerNormMatchesKnownMeanAndVariance) {
    aicore::depth::Backend be;
    aicore::depth::GraphInputPool pool;
    // [1,2,3,4] has mean=2.5 and variance=1.25; w=1,b=0.
    std::vector<float> x = {1, 2, 3, 4}, w = {1, 1, 1, 1}, b = {0, 0, 0, 0},
                       out;
    ASSERT_TRUE(be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* tx = be.add_graph_input(ctx, pool, x.data(), 4);
                ggml_tensor* tw = be.add_graph_input(ctx, pool, w.data(), 4);
                ggml_tensor* tb = be.add_graph_input(ctx, pool, b.data(), 4);
                return ggml_common::layernorm(ctx, tx, tw, tb, 1e-6f);
            },
            out));
    ASSERT_EQ(out.size(), x.size());
    const float sd = std::sqrt(1.25f);
    EXPECT_LT(std::fabs(out[0] - (-1.5f / sd)), 1e-3f);
    EXPECT_LT(std::fabs(out[3] - (1.5f / sd)), 1e-3f);
}

TEST(DepthGgmlExtend, GeluUsesExactErfRatherThanTanhApproximation) {
    aicore::depth::Backend be;
    aicore::depth::GraphInputPool pool;
    std::vector<float> x = {1.0f, -1.0f, 0.5f, 2.0f}, out;
    ASSERT_TRUE(be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* t =
                        be.add_graph_input(ctx, pool, x.data(), x.size());
                return ggml_common::gelu_erf(ctx, t);
            },
            out));
    ASSERT_EQ(out.size(), x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        const float ref =
                x[i] * 0.5f * (1.0f + std::erf(x[i] / std::sqrt(2.0f)));
        EXPECT_LT(std::fabs(out[i] - ref), 1e-3f) << "element " << i;
    }
}
