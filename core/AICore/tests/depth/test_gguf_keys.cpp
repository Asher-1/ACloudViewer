// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstring>

#include "depth_gguf_keys.h"

static int failures = 0;
#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                         \
            failures++;                                                  \
        }                                                                \
    } while (0)

int main() {
    CHECK(std::strcmp(AICORE_DEPTH_ARCH, "depthanything3") == 0);
    CHECK(std::strcmp(AICORE_DEPTH_KV_VIT_EMBED_DIM,
                      "depthanything3.vit.embed_dim") == 0);
    CHECK(std::strcmp(AICORE_DEPTH_KV_HEAD_MAX_DEPTH,
                      "depthanything3.head.max_depth") == 0);
    return failures == 0 ? 0 : 1;
}
