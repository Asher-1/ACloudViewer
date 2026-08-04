// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/deeplsd_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_deeplsd_abi_version() >= 1);

    aicore_deeplsd_free(nullptr);
    aicore_deeplsd_options_free(nullptr);
    aicore_deeplsd_free_string(nullptr);

    AICORE_CHECK(aicore_deeplsd_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_deeplsd_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_deeplsd_info_json(nullptr) != nullptr);

    aicore_deeplsd_options* opts = aicore_deeplsd_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_deeplsd_options_set_device(opts, "cpu");
    aicore_deeplsd_options_set_threads(opts, 1);
    aicore_deeplsd_options_free(opts);

    float* df = nullptr;
    float* ang = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    AICORE_CHECK(aicore_deeplsd_extract_gray(nullptr, nullptr, 0, 0, 0, &df,
                                             &ang, &w, &h) != 0);

    AICORE_CHECK(aicore_deeplsd_quantize(nullptr, nullptr, nullptr) != 0);
    AICORE_CHECK(aicore_deeplsd_warmup_backend("cpu") == 0);

    char* dir = aicore_deeplsd_model_cache_dir();
    AICORE_CHECK(dir != nullptr && std::strlen(dir) > 0);
    aicore_deeplsd_free_string(dir);

    return failures;
}
