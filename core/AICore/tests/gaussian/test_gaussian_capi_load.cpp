// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>

#include "aicore/gaussian_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    const char* gguf = std::getenv("AICORE_TEST_GAUSSIAN_GGUF");
    if (!gguf || gguf[0] == '\0') return 77;

    aicore_gaussian_ctx* ctx = aicore_gaussian_load(gguf, 1);
    AICORE_CHECK(ctx != nullptr);

    aicore_gaussian_geometry geo{};
    AICORE_CHECK(aicore_gaussian_geometry_of(ctx, &geo) == 0);
    AICORE_CHECK(geo.image_height > 0 && geo.gaussian_channels > 0);

    char* json = aicore_gaussian_info_json(ctx);
    AICORE_CHECK(json != nullptr &&
                 std::strstr(json, "architecture") != nullptr);
    aicore_gaussian_free_string(json);

    aicore_gaussian_free(ctx);
    std::fprintf(stderr, "gaussian load ok: %s\n", gguf);
    return failures == 0 ? 0 : 1;
}
