// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Contract tests for aicore_aliked_* (no GGUF required).

#include <cstring>
#include <filesystem>

#include "aicore/aliked_capi.h"
#include "aicore/lightglue_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_aliked_abi_version() >= 1);

    aicore_aliked_free(nullptr);
    aicore_aliked_options_free(nullptr);
    aicore_aliked_free_string(nullptr);

    AICORE_CHECK(aicore_aliked_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_aliked_info_json(nullptr) != nullptr);

    aicore_aliked_options* opts = aicore_aliked_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_aliked_options_set_device(opts, "cpu");
    aicore_aliked_options_set_threads(opts, 1);
    aicore_aliked_options_set_max_keypoints(opts, 512);
    aicore_aliked_options_set_resize_long_edge(opts, 1024);
    aicore_aliked_options_free(opts);

    aicore_lightglue_features features{};
    AICORE_CHECK(aicore_aliked_extract_rgb(nullptr, nullptr, 0, 0, 0,
                                           &features) != 0);

    char* dir = aicore_aliked_model_cache_dir();
    AICORE_CHECK(dir != nullptr);
    AICORE_CHECK(std::strlen(dir) > 0);
    aicore_aliked_free_string(dir);

    return failures;
}
