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
#include <random>
#include <vector>

#include "tasks/depth/backend.hpp"
#include "tasks/depth/winograd.hpp"
#include "tests/depth/whitebox/parity.hpp"

int main() {
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

    if (!ok1 || !ok2) {
        std::fprintf(stderr, "[winograd] compute failed\n");
        return 1;
    }

    bool ok = da_parity::compare(got_wino, got_direct, "winograd_vs_direct",
                                 4e-2f, 3e-3f);
    return ok ? 0 : 1;
}
