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
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (!device || device[0] == '\0') device = "cpu";

    aicore_gaussian_options* opts = aicore_gaussian_options_new();
    AICORE_CHECK(opts != nullptr);
    if (!opts) return 1;
    aicore_gaussian_options_set_device(opts, device);
    aicore_gaussian_options_set_threads(opts, 0);
    aicore_gaussian_ctx* ctx = aicore_gaussian_load_opts(gguf, opts);
    aicore_gaussian_options_free(opts);
    AICORE_CHECK(ctx != nullptr);
    if (!ctx) return 1;

    aicore_gaussian_geometry geo{};
    AICORE_CHECK(aicore_gaussian_geometry_of(ctx, &geo) == 0);
    AICORE_CHECK(geo.image_height > 0 && geo.gaussian_channels > 0);

    char* json = aicore_gaussian_info_json(ctx);
    AICORE_CHECK(json != nullptr &&
                 std::strstr(json, "architecture") != nullptr);
    aicore_gaussian_free_string(json);

    const char* image0 = std::getenv("AICORE_TEST_GAUSSIAN_IMAGE_0");
    const char* image1 = std::getenv("AICORE_TEST_GAUSSIAN_IMAGE_1");
    if (image0 && image0[0] && image1 && image1[0]) {
        const char* paths[] = {image0, image1};
        float* output = nullptr;
        size_t output_count = 0;
        AICORE_CHECK(aicore_gaussian_run_paths(
                             ctx, paths, 2, &output, &output_count) == 0);
        AICORE_CHECK(output != nullptr && output_count > 0);
        aicore_gaussian_free_floats(output);
        std::fprintf(stderr,
                     "gaussian inference ok: device=%s output=%zu floats\n",
                     device, output_count);
    }

    aicore_gaussian_free(ctx);
    std::fprintf(stderr, "gaussian load ok: %s\n", gguf);
    return failures == 0 ? 0 : 1;
}
