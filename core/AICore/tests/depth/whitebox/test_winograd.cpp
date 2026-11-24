// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Correctness gate: Winograd F(2x2,3x3) vs ggml_conv_2d_direct on a random
// [3,3,IC,OC] filter and [W,H,IC,N] input. F(2x2,3x3) is numerically exact
// (transforms are halves/integers); the only difference vs the reference is
// the f32 accumulation ORDER (blocked winograd-domain GEMM vs direct conv).
// A 576-term (3x3x64) dot with |terms|~1 has |y|~8..34, so order-noise is
// ~eps*sqrt(K)*|y| ~ 1e-4 relative (~3e-3 absolute at the tails) and near-zero
// outputs suffer cancellation: the absolute floor must scale with the OUTPUT
// magnitude, not the input magnitude. atol 4e-2 + rtol 3e-3 bounds that noise
// with >4x margin (measured: max|d| 2.8e-2 on the |y|<=34 tail).
#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <vector>

#include "tasks/depth/backend.hpp"
#include "tasks/depth/winograd.hpp"

TEST(DepthWinograd, MatchesDirectConvolutionWithinAccumulationTolerance) {
    const int W = 128, H = 96, IC = 64, OC = 64, N = 1, pad = 1;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> xin((size_t)W * H * IC * N);
    std::vector<float> win((size_t)3 * 3 * IC * OC);
    for (auto& v : xin) v = dist(rng);
    for (auto& v : win) v = dist(rng);

    aicore::depth::Backend be;

    auto build_input = [&](ggml_context* ctx,
                           aicore::depth::GraphInputPool& pool, ggml_tensor** x,
                           ggml_tensor** w) {
        const int64_t xne[4] = {W, H, IC, N};
        const int64_t wne[4] = {3, 3, IC, OC};
        *x = be.add_graph_input_nd(ctx, pool, xin.data(), xne, 4);
        *w = be.add_graph_input_nd(ctx, pool, win.data(), wne, 4);
    };

    aicore::depth::GraphInputPool pool_w, pool_d;
    std::vector<float> got_wino, got_direct;

    bool ok1 = be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor *x, *w;
                build_input(ctx, pool_w, &x, &w);
                return aicore::depth::winograd_conv3x3(ctx, w, x, pad);
            },
            got_wino);

    bool ok2 = be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor *x, *w;
                build_input(ctx, pool_d, &x, &w);
                return ggml_conv_2d_direct(ctx, w, x, 1, 1, pad, pad, 1, 1);
            },
            got_direct);

    ASSERT_TRUE(ok1) << "Winograd graph computation failed";
    ASSERT_TRUE(ok2) << "Direct convolution graph computation failed";
    ASSERT_FALSE(got_wino.empty());
    ASSERT_EQ(got_wino.size(), got_direct.size());
    for (size_t i = 0; i < got_wino.size(); ++i) {
        const double tolerance =
                static_cast<double>(4e-2f) +
                static_cast<double>(3e-3f) *
                        std::fabs(static_cast<double>(got_direct[i]));
        ASSERT_LE(std::fabs(static_cast<double>(got_wino[i]) - got_direct[i]),
                  tolerance)
                << "element " << i << ": Winograd=" << got_wino[i]
                << ", direct=" << got_direct[i];
    }
}
