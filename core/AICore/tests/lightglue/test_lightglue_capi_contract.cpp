// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Contract tests for aicore_lightglue_* (no GGUF required).

#include <cstring>
#include <filesystem>

#include "aicore/lightglue_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_lightglue_abi_version() >= 1);

    aicore_lightglue_free(nullptr);
    aicore_lightglue_free_matches(nullptr);
    aicore_lightglue_free_string(nullptr);
    aicore_lightglue_free_features(nullptr);
    aicore_lightglue_options_free(nullptr);

    AICORE_CHECK(aicore_lightglue_load(nullptr, 1) == nullptr);
    AICORE_CHECK(aicore_lightglue_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_lightglue_info_json(nullptr) == nullptr);

    aicore_lightglue_options* opts = aicore_lightglue_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_lightglue_options_set_device(opts, "cpu");
    aicore_lightglue_options_set_threads(opts, 1);
    aicore_lightglue_options_set_min_score(opts, 0.1);
    aicore_lightglue_options_set_matcher_type(opts, 2);
    aicore_lightglue_options_free(opts);

    aicore_lightglue_geometry geo{};
    AICORE_CHECK(aicore_lightglue_geometry_of(nullptr, &geo) != 0);

    aicore_lightglue_match* matches = nullptr;
    int32_t n_matches = 0;
    AICORE_CHECK(aicore_lightglue_run_match(nullptr, nullptr, nullptr, &matches,
                                            &n_matches) != 0);

    aicore_lightglue_features f0{}, f1{};
    AICORE_CHECK(aicore_lightglue_load_fixture("/nonexistent.bin", &f0, &f1) !=
                 0);

    AICORE_CHECK(aicore_lightglue_quantize(nullptr, nullptr, nullptr) != 0);

    AICORE_CHECK(aicore_lightglue_warmup_backend("cpu") == 0);

    char* dir = aicore_lightglue_model_cache_dir();
    AICORE_CHECK(dir != nullptr);
    AICORE_CHECK(std::strlen(dir) > 0);
    aicore_lightglue_free_string(dir);

    return failures;
}
