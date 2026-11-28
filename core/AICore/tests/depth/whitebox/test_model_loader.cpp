// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "tasks/depth/model_loader.hpp"

TEST(DepthWhitebox, ModelLoader) {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    if (!gguf) {
        std::fprintf(stderr, "AICORE_TEST_DEPTH_GGUF unset; skipping\n");
        GTEST_SKIP() << "AICORE_TEST_DEPTH_GGUF is unset";
    }
    aicore::depth::ModelLoader ml;
    if (!ml.load(gguf)) {
        std::fprintf(stderr, "load failed\n");
        FAIL() << "Model operation failed";
    }
    const auto& c = ml.config();
    EXPECT_EQ(c.embed_dim, 768u);
    EXPECT_EQ(c.depth, 12u);
    EXPECT_EQ(c.num_heads, 12u);
    EXPECT_EQ(c.head_dim, 64u);
    EXPECT_EQ(c.alt_start, 4u);
    EXPECT_EQ(c.rope_start, 4u);
    EXPECT_EQ(c.qknorm_start, 4u);
    EXPECT_EQ(c.out_layers.size(), 4u);
    EXPECT_NE(ml.tensor("vit.patch_embed.weight"), nullptr);
    EXPECT_NE(ml.tensor("vit.blk.0.attn_qkv.weight"), nullptr);
    EXPECT_NE(ml.tensor("vit.blk.11.mlp_fc2.weight"), nullptr);
}
