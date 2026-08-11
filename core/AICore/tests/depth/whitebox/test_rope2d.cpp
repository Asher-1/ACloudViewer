// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <vector>

#include "backend.hpp"
#include "parity.hpp"
#include "rope2d.hpp"
int main() {
    const char* base = std::getenv("AICORE_TEST_DEPTH_BASELINE");
    if (!base) return 77;
    std::vector<float> rin, rpos, rout;
    std::vector<int64_t> s;
    if (!da_parity::load_baseline(base, "rope_in", rin, s)) return 77;
    da_parity::load_baseline(base, "rope_pos", rpos, s);
    da_parity::load_baseline(base, "rope_out", rout, s);
    const int hd = 64, tok = 4;
    aicore::depth::RopeTables rt =
            aicore::depth::build_rope_tables(rpos, tok, hd, 100.f);
    aicore::depth::Backend be;
    aicore::depth::GraphInputPool pool;
    std::vector<float> got;
    be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                const int64_t ne[3] = {hd, 1, tok};
                ggml_tensor* x =
                        be.add_graph_input_nd(ctx, pool, rin.data(), ne, 3);
                ggml_tensor *cb, *sb;
                aicore::depth::build_rope_inputs(ctx, be, pool, rt, cb, sb);
                return aicore::depth::apply_rope(ctx, x, cb, sb, hd);
            },
            got);
    bool ok = da_parity::compare(got, rout, "rope2d", 1e-4f, 1e-4f);
    return ok ? 0 : 1;
}
