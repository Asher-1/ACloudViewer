// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// MapAnything model-free public contract: catalog metadata, invalid devices
// and empty-context access.
#include <cstring>

#include "aicore/map_anything_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_map_anything_abi_version() >= 1);
    AICORE_CHECK(aicore_map_anything_model_count() > 0);
    AICORE_CHECK(aicore_map_anything_model_by_filename(nullptr) == nullptr);
    AICORE_CHECK(aicore_map_anything_model_by_filename("not-a-model.gguf") ==
                 nullptr);
    AICORE_CHECK(aicore_map_anything_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_map_anything_model_at(
                         aicore_map_anything_model_count()) == nullptr);
    const char* base = aicore_map_anything_model_download_base();
    AICORE_CHECK(base && std::strncmp(base, "https://", 8) == 0);
    for (int i = 0; i < aicore_map_anything_model_count(); ++i) {
        const auto* entry = aicore_map_anything_model_at(i);
        AICORE_CHECK(entry && entry->filename && entry->download_url);
        if (!entry || !entry->filename) continue;
        AICORE_CHECK(aicore_map_anything_model_by_filename(entry->filename) ==
                     entry);
        if (entry->download_url && base)
            AICORE_CHECK(std::strncmp(entry->download_url, base,
                                      std::strlen(base)) == 0);
    }
    AICORE_CHECK(aicore_map_anything_warmup_backend("not-a-real-backend") != 0);
    aicore_map_anything_options_set_log_level(nullptr, 3);
    auto* opts = aicore_map_anything_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_map_anything_options_set_log_level(opts, 3);
    aicore_map_anything_options_set_device(opts, "cpu");
    AICORE_CHECK(aicore_map_anything_load_opts(nullptr, opts) == nullptr);
    aicore_map_anything_options_free(opts);
    AICORE_CHECK(aicore_map_anything_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_map_anything_run_images(nullptr, nullptr, 0) == -1);
    AICORE_CHECK(aicore_map_anything_result_view_count(nullptr) == 0);
    AICORE_CHECK(aicore_map_anything_result_depth(nullptr) == nullptr);
    aicore_map_anything_free(nullptr);
    aicore_map_anything_free_buffer(nullptr);
    aicore_map_anything_shutdown();
    aicore_map_anything_shutdown();
    return failures ? 1 : 0;
}
