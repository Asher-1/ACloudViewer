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
    AICORE_CHECK(aicore_rmbg_abi_version() >= 2);

    // Null-safe teardown / lifecycle.
    aicore_rmbg_free(nullptr);
    aicore_rmbg_options_free(nullptr);
    aicore_rmbg_free_string(nullptr);
    aicore_rmbg_free_buffer(nullptr);

    AICORE_CHECK(aicore_rmbg_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_rmbg_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_rmbg_last_error(nullptr) == nullptr);
    aicore_rmbg_timings timings{};
    AICORE_CHECK(aicore_rmbg_last_timings(nullptr, &timings) == -1);
    AICORE_CHECK(aicore_rmbg_last_timings(nullptr, nullptr) == -1);

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
    AICORE_CHECK(aicore_rmbg_last_timings(ctx, &timings) == -1);

    uint8_t* png = nullptr;
    int png_len = 0;
    AICORE_CHECK(aicore_rmbg_remove_background_path(
                         ctx, "/nonexistent/image.png", &png, &png_len) == -1);
    AICORE_CHECK(png == nullptr && png_len == 0);

    static const uint8_t kRgb[3 * 3 * 3] = {
            255, 0,   0, 0,   255, 0,   0,   0,  255, 255, 255, 0,  0, 255,
            255, 255, 0, 255, 128, 128, 128, 64, 64,  64,  32,  32, 32};
    AICORE_CHECK(aicore_rmbg_remove_background_rgb(ctx, kRgb, 3, 3, &png,
                                                   &png_len) == -1);

    uint8_t* alpha = nullptr;
    int32_t aw = 0;
    int32_t ah = 0;
    AICORE_CHECK(aicore_rmbg_alpha_mat_rgb(ctx, kRgb, 3, 3, &alpha, &aw, &ah) ==
                 -1);

    // Raw-RGBA variant: same failure contract as the PNG path, plus output
    // pointers must be zeroed on failure.
    uint8_t* rgba = nullptr;
    int32_t rw = 0, rh = 0;
    int rlen = 0;
    AICORE_CHECK(aicore_rmbg_remove_background_rgba(ctx, kRgb, 3, 3, &rgba, &rw,
                                                    &rh, &rlen) == -1);
    AICORE_CHECK(rgba == nullptr && rw == 0 && rh == 0 && rlen == 0);
    aicore_rmbg_free(ctx);
    aicore_rmbg_options_free(opts);

    // Null guards on inference entry points.
    AICORE_CHECK(aicore_rmbg_remove_background_rgb(nullptr, kRgb, 3, 3, &png,
                                                   &png_len) == -1);
    AICORE_CHECK(aicore_rmbg_remove_background_rgb(ctx /* null */, nullptr, 0,
                                                   0, &png, &png_len) == -1);
    AICORE_CHECK(aicore_rmbg_alpha_mat_rgb(nullptr, kRgb, 3, 3, &alpha, &aw,
                                           &ah) == -1);
    AICORE_CHECK(aicore_rmbg_remove_background_rgba(nullptr, kRgb, 3, 3, &rgba,
                                                    &rw, &rh, &rlen) == -1);
    // Both pointers null: the parameter guard fires on rgb before any ctx
    // dereference (pass nullptr explicitly — the ctx above is already freed).
    AICORE_CHECK(aicore_rmbg_remove_background_rgba(
                         nullptr, nullptr, 0, 0, &rgba, &rw, &rh, &rlen) == -1);

    // Model catalog contract.
    // 3 quantized variants: f32, f16, q8 (must match the trellis2-ggml
    // release assets; rmbg_q8_0.gguf / rmbg_q4_K.gguf do not exist).
    AICORE_CHECK(aicore_rmbg_model_count() == 3);
    static const char* kExpectedFilenames[] = {"rmbg_f32.gguf", "rmbg_f16.gguf",
                                               "rmbg_q8.gguf"};
    for (int i = 0; i < 3; ++i) {
        const aicore_rmbg_model_entry* e = aicore_rmbg_model_at(i);
        AICORE_CHECK(e != nullptr && e->filename != nullptr &&
                     std::strcmp(e->filename, kExpectedFilenames[i]) == 0 &&
                     e->download_url != nullptr && e->display_name != nullptr &&
                     e->quant_note != nullptr && e->license_note != nullptr);
    }
    AICORE_CHECK(aicore_rmbg_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_rmbg_model_at(3) == nullptr);
    AICORE_CHECK(aicore_rmbg_model_by_filename("rmbg_f16.gguf") != nullptr);
    AICORE_CHECK(aicore_rmbg_model_by_filename("rmbg_f32.gguf") != nullptr);
    AICORE_CHECK(aicore_rmbg_model_by_filename("rmbg_q8.gguf") != nullptr);
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
