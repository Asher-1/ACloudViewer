// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include "tasks/depth/depth_gguf_keys.h"

TEST(DepthGgufKeys, ArchitectureMatchesConverterContract) {
    EXPECT_STREQ(AICORE_DEPTH_ARCH, "depthanything3");
}

TEST(DepthGgufKeys, MetadataNamesMatchConverterContract) {
    EXPECT_STREQ(AICORE_DEPTH_KV_VIT_EMBED_DIM, "depthanything3.vit.embed_dim");
    EXPECT_STREQ(AICORE_DEPTH_KV_HEAD_MAX_DEPTH,
                 "depthanything3.head.max_depth");
}
