// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/eloftr_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_eloftr_abi_version() >= 1);

    aicore_eloftr_free(nullptr);
    aicore_eloftr_options_free(nullptr);
    aicore_eloftr_free_string(nullptr);
    aicore_eloftr_free_matches(nullptr);

    AICORE_CHECK(aicore_eloftr_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_eloftr_info_json(nullptr) != nullptr);

    aicore_eloftr_options* opts = aicore_eloftr_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_eloftr_options_set_device(opts, "cpu");
    aicore_eloftr_options_set_threads(opts, 1);
    aicore_eloftr_options_free(opts);

    aicore_eloftr_match* matches = nullptr;
    int32_t count = 0;
    AICORE_CHECK(aicore_eloftr_match_gray(nullptr, nullptr, nullptr, 0, 0, 0,
                                          &matches, &count) != 0);

    AICORE_CHECK(aicore_eloftr_quantize(nullptr, nullptr, nullptr) != 0);
    AICORE_CHECK(aicore_eloftr_warmup_backend("cpu") == 0);

    char* dir = aicore_eloftr_model_cache_dir();
    AICORE_CHECK(dir != nullptr && std::strlen(dir) > 0);
    aicore_eloftr_free_string(dir);

    return failures;
}
