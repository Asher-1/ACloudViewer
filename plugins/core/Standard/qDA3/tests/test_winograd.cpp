// Correctness gate: Winograd F(2x2,3x3) vs ggml_conv_2d_direct on a random
// [3,3,IC,OC] filter and [W,H,IC,N] input. F(2x2,3x3) is numerically exact
// (transforms are halves/integers), so max|d| should be ~1e-4..1e-3 << 2e-3.
#include "winograd.hpp"
#include "backend.hpp"
#include "parity.hpp"
#include <random>
#include <vector>

int main() {
    const int W = 128, H = 96, IC = 64, OC = 64, N = 1, pad = 1;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> xin((size_t)W * H * IC * N);
    std::vector<float> win((size_t)3 * 3 * IC * OC);
    for (auto& v : xin) v = dist(rng);
    for (auto& v : win) v = dist(rng);

    da::Backend be;

    auto build_input = [&](ggml_context* ctx, da::GraphInputPool& pool,
                           ggml_tensor** x, ggml_tensor** w) {
        const int64_t xne[4] = { W, H, IC, N };
        const int64_t wne[4] = { 3, 3, IC, OC };
        *x = be.add_graph_input_nd(ctx, pool, xin.data(), xne, 4);
        *w = be.add_graph_input_nd(ctx, pool, win.data(), wne, 4);
    };

    da::GraphInputPool pool_w, pool_d;
    std::vector<float> got_wino, got_direct;

    bool ok1 = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor *x, *w; build_input(ctx, pool_w, &x, &w);
        return da::winograd_conv3x3(ctx, w, x, pad);
    }, got_wino);

    bool ok2 = be.compute([&](ggml_context* ctx) -> ggml_tensor* {
        ggml_tensor *x, *w; build_input(ctx, pool_d, &x, &w);
        return ggml_conv_2d_direct(ctx, w, x, 1, 1, pad, pad, 1, 1);
    }, got_direct);

    if (!ok1 || !ok2) { std::fprintf(stderr, "[winograd] compute failed\n"); return 1; }

    bool ok = da_parity::compare(got_wino, got_direct, "winograd_vs_direct", 2e-3f, 2e-3f);
    return ok ? 0 : 1;
}
