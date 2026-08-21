// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// RF-DETR C API contract test — fast, no GGUF assets required. Covers ABI,
// options lifecycle, error paths, model catalog and device enumeration.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/rfdetr_capi.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_rfdetr_abi_version() >= 1);

    // Null-safe teardown / lifecycle.
    aicore_rfdetr_free(nullptr);
    aicore_rfdetr_options_free(nullptr);
    aicore_rfdetr_free_buffer(nullptr);
    aicore_rfdetr_free_buffer(nullptr);

    AICORE_CHECK(aicore_rfdetr_load_opts(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_rfdetr_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_rfdetr_last_error(nullptr) == nullptr);
    AICORE_CHECK(aicore_rfdetr_detection_count(nullptr) == -1);
    AICORE_CHECK(aicore_rfdetr_detection_mask(nullptr, 0, nullptr, 0, nullptr,
                                              nullptr) == -1);
    AICORE_CHECK(aicore_rfdetr_detection_mask_png(nullptr, 0, nullptr, 0) ==
                 -1);

    // Options lifecycle.
    aicore_rfdetr_options* opts = aicore_rfdetr_options_new();
    AICORE_CHECK(opts != nullptr);
    aicore_rfdetr_options_set_device(opts, "cpu");
    aicore_rfdetr_options_set_threads(opts, 1);

    // Loading a nonexistent file must fail cleanly and report an error.
    aicore_rfdetr_ctx* ctx =
            aicore_rfdetr_load_opts("/nonexistent/rfdetr-model.gguf", opts);
    AICORE_CHECK(ctx != nullptr);  // ctx is allocated; engine inside is null
    AICORE_CHECK(aicore_rfdetr_is_ready(ctx) == 0);
    AICORE_CHECK(aicore_rfdetr_last_error(ctx) != nullptr);
    AICORE_CHECK(aicore_rfdetr_detect_path_json(ctx, "/nonexistent/image.png",
                                                0.5f, 300) == nullptr);
    aicore_rfdetr_free(ctx);
    aicore_rfdetr_options_free(opts);

    // RGB-buffer entry points reject invalid input.
    AICORE_CHECK(aicore_rfdetr_detect_rgb_json(nullptr, nullptr, 0, 0, 0.5f,
                                               300) == nullptr);

    uint8_t* rgb = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    AICORE_CHECK(aicore_rfdetr_load_path_rgb("/nonexistent/image.png", &rgb, &w,
                                             &h) == -1);

    // Model catalog contract.
    // 44 models = 11 variants (5 detection + 6 segmentation) x 4 quants
    // (f32, f16, q8_0, q4_K).
    AICORE_CHECK(aicore_rfdetr_model_count() == 44);
    AICORE_CHECK(aicore_rfdetr_detection_model_count() == 20);
    AICORE_CHECK(aicore_rfdetr_segmentation_model_count() == 24);
    AICORE_CHECK(aicore_rfdetr_detection_model_count() +
                         aicore_rfdetr_segmentation_model_count() ==
                 aicore_rfdetr_model_count());
    const aicore_rfdetr_model_entry* first = aicore_rfdetr_model_at(0);
    AICORE_CHECK(first != nullptr && first->filename != nullptr &&
                 first->download_url != nullptr);
    AICORE_CHECK(aicore_rfdetr_model_at(-1) == nullptr);
    AICORE_CHECK(aicore_rfdetr_model_at(aicore_rfdetr_model_count()) ==
                 nullptr);
    const aicore_rfdetr_model_entry* by_name =
            aicore_rfdetr_model_by_filename(first->filename);
    AICORE_CHECK(by_name != nullptr &&
                 std::strcmp(by_name->download_url, first->download_url) == 0);
    AICORE_CHECK(aicore_rfdetr_model_by_filename("nope.gguf") == nullptr);
    AICORE_CHECK(aicore_rfdetr_model_download_base() != nullptr &&
                 std::strstr(aicore_rfdetr_model_download_base(),
                             "RF-DETR-GGUF") != nullptr);

    // Role-tagged catalog accessors.
    const aicore_rfdetr_model_entry* det0 = aicore_rfdetr_detection_model_at(0);
    AICORE_CHECK(det0 != nullptr && det0->segmentation_capable == 0);
    AICORE_CHECK(aicore_rfdetr_detection_model_at(20) == nullptr);
    const aicore_rfdetr_model_entry* seg0 =
            aicore_rfdetr_segmentation_model_at(0);
    AICORE_CHECK(seg0 != nullptr && seg0->segmentation_capable == 1);
    AICORE_CHECK(aicore_rfdetr_segmentation_model_at(24) == nullptr);

    // Model-free introspection on a ctx with no loaded model.
    aicore_rfdetr_ctx* empty_ctx =
            aicore_rfdetr_load_opts("/nonexistent/rfdetr.gguf", nullptr);
    AICORE_CHECK(empty_ctx != nullptr);
    AICORE_CHECK(aicore_rfdetr_is_ready(empty_ctx) == 0);
    AICORE_CHECK(aicore_rfdetr_context_variant(empty_ctx) != nullptr);
    AICORE_CHECK(aicore_rfdetr_context_image_size(empty_ctx) == 0);
    AICORE_CHECK(aicore_rfdetr_context_num_classes(empty_ctx) == 0);
    AICORE_CHECK(aicore_rfdetr_context_has_segmentation(empty_ctx) == 0);
    AICORE_CHECK(aicore_rfdetr_context_device(empty_ctx) != nullptr);
    // No model loaded -> no engine -> threads must read as 0.
    AICORE_CHECK(aicore_rfdetr_context_threads(empty_ctx) == 0);
    // info_json returns NULL without a loaded model (no metadata to serialize);
    // free_string must accept NULL.
    char* info = aicore_rfdetr_info_json(empty_ctx);
    aicore_rfdetr_free_buffer(info);
    aicore_rfdetr_free(empty_ctx);

    // Device enumeration / warmup.
    AICORE_CHECK(aicore_rfdetr_warmup_backend("cpu") == 0);
    aicore_rfdetr_shutdown();
    aicore_rfdetr_shutdown();  // idempotent

    char* dir = aicore_rfdetr_model_cache_dir();
    AICORE_CHECK(dir != nullptr && std::strlen(dir) > 0);
    aicore_rfdetr_free_buffer(dir);

    if (failures == 0) {
        std::printf("[rfdetr] contract test passed\n");
    }
    return failures;
}
