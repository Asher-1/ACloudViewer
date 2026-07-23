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
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (!device || device[0] == '\0') device = "cpu";

    aicore_depth_ctx* ctx = aicore_depth_load_device(gguf, 0, device);
    AICORE_CHECK(ctx != nullptr);
    if (!ctx) return 1;

    char* json = aicore_depth_info_json(ctx);
    AICORE_CHECK(json != nullptr && std::strstr(json, "embed_dim") != nullptr);
    aicore_depth_free_string(json);

    const char* image = std::getenv("AICORE_TEST_DEPTH_IMAGE");
    if (image && image[0]) {
        aicore_depth_set_img_resize_target(ctx, 224);
        int height = 0;
        int width = 0;
        float* depth = aicore_depth_depth_path(ctx, image, &height, &width);
        AICORE_CHECK(depth != nullptr);
        AICORE_CHECK(height > 0 && width > 0);
        aicore_depth_free_floats(depth);
        std::fprintf(stderr,
                     "depth inference ok: device=%s size=%dx%d\n", device,
                     width, height);
    }

    aicore_depth_release_gpu_working_memory(ctx);
    aicore_depth_free(ctx);
    std::fprintf(stderr, "depth load ok: %s\n", gguf);
    return failures == 0 ? 0 : 1;
}
