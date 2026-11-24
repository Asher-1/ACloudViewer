// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <vector>

#include "ggml.h"
#include "tasks/depth/backend.hpp"

TEST(DepthBackend, ComputesElementwiseAddition) {
    aicore::depth::Backend be;
    aicore::depth::GraphInputPool pool;
    std::vector<float> a = {1, 2, 3, 4}, b = {10, 20, 30, 40}, out;
    ASSERT_TRUE(be.compute(
            [&](ggml_context* ctx) -> ggml_tensor* {
                ggml_tensor* ta =
                        be.add_graph_input(ctx, pool, a.data(), a.size());
                ggml_tensor* tb =
                        be.add_graph_input(ctx, pool, b.data(), b.size());
                return ggml_add(ctx, ta, tb);
            },
            out));
    ASSERT_EQ(out.size(), 4u);
    EXPECT_EQ(out[0], 11);
    EXPECT_EQ(out[3], 44);
}
