// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// RMBG-2.0 C API contract test — fast, no GGUF assets required. Covers ABI,
// options lifecycle, error paths, model catalog and device enumeration.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/rmbg_capi.h"
#include "common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_rmbg_abi_version() >= 1);

    // Null-safe teardown / lifecycle.
    aicore_rmbg_free(nullptr);
    aicore_rmbg_options_free(nullptr);
    aicore_rmbg_free_string(nullptr);
    aicore_rmbg_free_buffer(nullptr);

    AICORE_CHECK(aicore_rmbg_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_rmbg_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_rmbg_last_error(nullptr) == nullptr);

    // Options lifecycle.
    aicore_rmbg_options* opts = aicore_rmbg_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_rmbg_options_set_device(opts, "cpu");
    aicore_rmbg_options_set_threads(opts, 1);

    // Loading a nonexistent file must fail cleanly and report an error.
    aicore_rmbg_ctx* ctx =
            aicore_rmbg_load_opts("/nonexistent/rmbg.gguf", opts);
    AICORE_CHECK(ctx != nullptr);  // ctx is allocated; model inside is null
    AICORE_CHECK(aicore_rmbg_is_ready(ctx) == 0);
    AICORE_CHECK(aicore_rmbg_last_error(ctx) != nullptr);

    uint8_t* png = nullptr;
    int png_len = 0;
    AICORE_CHECK(aicore_rmbg_remove_background_path(
                         ctx, "/nonexistent/image.png", &png, &png_len) == -1);
    AICORE_CHECK(png == nullptr && png_len == 0);

    static const uint8_t kRgb[3 * 3 * 3] = {255, 0,   0,   0,   255, 0,
                                            0,   0,   255, 255, 255, 0,
                                            0,   255, 255, 255, 0,   255,
                                            128, 128, 128, 64,  64,  64,
                                            32,  32,  32};
    AICORE_CHECK(aicore_rmbg_remove_background_rgb(
                         ctx, kRgb, 3, 3, &png, &png_len) == -1);

    uint8_t* alpha = nullptr;
    int32_t aw = 0;
    int32_t ah = 0;
    AICORE_CHECK(aicore_rmbg_alpha_mat_rgb(ctx, kRgb, 3, 3, &alpha, &aw,
                                           &ah) == -1);
    aicore_rmbg_free(ctx);
    aicore_rmbg_options_free(opts);

    // Null guards on inference entry points.
    AICORE_CHECK(aicore_rmbg_remove_background_rgb(
                         nullptr, kRgb, 3, 3, &png, &png_len) == -1);
    AICORE_CHECK(aicore_rmbg_remove_background_rgb(
                         ctx /* null */, nullptr, 0, 0, &png, &png_len) ==
                 -1);
    AICORE_CHECK(aicore_rmbg_alpha_mat_rgb(nullptr, kRgb, 3, 3, &alpha, &aw,
                                           &ah) == -1);

    // Model catalog contract.
    AICORE_CHECK(aicore_rmbg_model_count() == 1);
    const aicore_rmbg_model_entry* entry = aicore_rmbg_model_at(0);
    AICORE_CHECK(entry != nullptr && entry->filename != nullptr &&
                 std::strcmp(entry->filename, "rmbg_f16.gguf") == 0 &&
                 entry->download_url != nullptr);
    AICORE_CHECK(aicore_rmbg_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_rmbg_model_at(1) == nullptr);
    AICORE_CHECK(aicore_rmbg_model_by_filename("rmbg_f16.gguf") == entry);
    AICORE_CHECK(aicore_rmbg_model_by_filename("nope.gguf") == nullptr);
    AICORE_CHECK(aicore_rmbg_model_download_base() != nullptr &&
                 std::strstr(aicore_rmbg_model_download_base(),
                             "trellis2-ggml") != nullptr);

    // Device enumeration / warmup.
    AICORE_CHECK(aicore_rmbg_warmup_backend("cpu") == 0);

    char* dir = aicore_rmbg_model_cache_dir();
    AICORE_CHECK(dir != nullptr && std::strlen(dir) > 0);
    aicore_rmbg_free_string(dir);

    if (failures == 0) {
        std::printf("[rmbg] contract test passed\n");
    }
    return failures;
}
