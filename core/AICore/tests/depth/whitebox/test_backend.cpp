// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <vector>

#include "backend.hpp"
#include "ggml.h"
int main() {
    aicore::depth::Backend be;
    aicore::depth::GraphInputPool pool;
    std::vector<float> a = {1, 2, 3, 4}, b = {10, 20, 30, 40}, out;
    bool ok = be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* ta =
                        be.add_graph_input(ctx, pool, a.data(), a.size());
                ggml_tensor* tb =
                        be.add_graph_input(ctx, pool, b.data(), b.size());
                return ggml_add(ctx, ta, tb);
            },
            out);
    ok = ok && out.size() == 4 && out[0] == 11 && out[3] == 44;
    std::fprintf(stderr, "backend add -> %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
