// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Isolation gates for the riskiest DPT conv ops against real-weight fixtures:
//   - conv-transpose (head.resize.0): the key kernel-layout unknown
//   - strided conv   (head.resize.3): 3x3 stride2 pad1
//   - 1x1 conv       (head.proj.0)
#include <cstdlib>
#include <string>
#include <vector>

#include "backend.hpp"
#include "dpt_blocks.hpp"
#include "model_loader.hpp"
#include "parity.hpp"

using aicore::depth::conv2d;
using aicore::depth::conv_transpose2d_p0;

// Run a conv gate: load input fixture [W,H,IC,1], run `build_op`, compare to
// expected.
static bool
gate(aicore::depth::ModelLoader& /*ml*/,
     aicore::depth::Backend& be,
     const char* base,
     const char* in_key,
     const char* out_key,
     int W,
     int H,
     int IC,
     const std::function<ggml_tensor*(ggml_context*, ggml_tensor*)>& build_op,
     const char* label) {
    std::vector<float> in, expected;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, in_key, in, s)) return false;
    if (!da_parity::load_baseline(base, out_key, expected, s)) return false;

    aicore::depth::GraphInputPool pool;
    std::vector<float> got;
    bool ran = be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                const int64_t ine[4] = {W, H, IC, 1};
                ggml_tensor* x =
                        be.add_graph_input_nd(ctx, pool, in.data(), ine, 4);
                return build_op(ctx, x);
            },
            got);
    if (!ran) {
        std::fprintf(stderr, "[%s] compute failed\n", label);
        return false;
    }
    return da_parity::compare(got, expected, label, 1e-3f, 1e-3f);
}

int main() {
    const char* gguf = std::getenv("DA_TEST_GGUF");
    const char* base = std::getenv("DA_TEST_BASELINE");
    if (!gguf || !base) return 77;
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) return 1;
    aicore::depth::Backend be;

    bool ok = true;

    // conv-transpose k4 s4 p0: 96->96, [16,16,96,1] -> [64,64,96,1]
    ok &= gate(
            ml, be, base, "convt0_in", "convt0_out", 16, 16, 96,
            [&](ggml_context* ctx, ggml_tensor* x) {
                return conv_transpose2d_p0(
                        ctx, ml.tensor("head.resize.0.weight"),
                        ml.tensor("head.resize.0.bias"), x, 4);
            },
            "convt0");

    // conv k3 s2 p1: 768->768, [16,16,768,1] -> [8,8,768,1]
    ok &= gate(
            ml, be, base, "convs3_in", "convs3_out", 16, 16, 768,
            [&](ggml_context* ctx, ggml_tensor* x) {
                return conv2d(ctx, ml.tensor("head.resize.3.weight"),
                              ml.tensor("head.resize.3.bias"), x, 2, 1);
            },
            "convs3");

    // conv 1x1 s1 p0: 1536->96, [16,16,1536,1] -> [16,16,96,1]
    ok &= gate(
            ml, be, base, "proj0_in", "proj0_out", 16, 16, 1536,
            [&](ggml_context* ctx, ggml_tensor* x) {
                return conv2d(ctx, ml.tensor("head.proj.0.weight"),
                              ml.tensor("head.proj.0.bias"), x, 1, 0);
            },
            "proj0");

    return ok ? 0 : 1;
}
