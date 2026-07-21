// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>

#include "aicore/depth_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    if (!gguf || gguf[0] == '\0') return 77;

    aicore_depth_ctx* ctx = aicore_depth_load(gguf, 1);
    AICORE_CHECK(ctx != nullptr);

    char* json = aicore_depth_info_json(ctx);
    AICORE_CHECK(json != nullptr && std::strstr(json, "embed_dim") != nullptr);
    aicore_depth_free_string(json);

    aicore_depth_release_gpu_working_memory(ctx);
    aicore_depth_free(ctx);
    std::fprintf(stderr, "depth load ok: %s\n", gguf);
    return failures == 0 ? 0 : 1;
}
