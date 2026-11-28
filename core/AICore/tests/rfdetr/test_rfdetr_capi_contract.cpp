// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// RF-DETR C API contract test — fast, no GGUF assets required. Covers ABI,
// options lifecycle, error paths, model catalog and device enumeration.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "aicore/rfdetr_capi.h"

TEST(RfdetrCApiContract, AbiAndNullReleaseQueries) {
    EXPECT_GE(aicore_rfdetr_abi_version(), 1);

    // Null-safe teardown / lifecycle.
    aicore_rfdetr_free(nullptr);
    aicore_rfdetr_options_free(nullptr);
    aicore_rfdetr_free_buffer(nullptr);
    aicore_rfdetr_free_buffer(nullptr);

    EXPECT_EQ(aicore_rfdetr_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_rfdetr_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_rfdetr_last_error(nullptr), nullptr);
    EXPECT_EQ(aicore_rfdetr_detection_count(nullptr), -1);
    EXPECT_EQ(aicore_rfdetr_detection_mask(nullptr, 0, nullptr, 0, nullptr,
                                           nullptr),
              -1);
    EXPECT_EQ(aicore_rfdetr_detection_mask_png(nullptr, 0, nullptr, 0), -1);
}

TEST(RfdetrCApiContract, ClassFilterOptionsAndFailedLoadClassNames) {
    // Options lifecycle.
    aicore_rfdetr_options* opts = aicore_rfdetr_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_rfdetr_options_set_device(opts, "cpu");
    aicore_rfdetr_options_set_threads(opts, 1);

    // Class-filter allowlist: null-safe, copies the array, and must not
    // break loading (the filter is applied at post-process time only).
    aicore_rfdetr_options_set_class_filter(nullptr, nullptr, 0);
    aicore_rfdetr_options_set_class_filter(opts, nullptr, 0);
    const uint32_t some_ids[] = {0u, 2u, 59u};
    aicore_rfdetr_options_set_class_filter(opts, some_ids, 3);
    aicore_rfdetr_options_set_class_filter(opts, some_ids, 0);  // reset
    aicore_rfdetr_options_set_class_filter(opts, some_ids, 3);

    // Class names: no model loaded -> NULL and count 0 (both via NULL ctx
    // and via a ctx whose engine failed to load).
    uint32_t n_names = 99u;
    EXPECT_EQ(aicore_rfdetr_context_class_names(nullptr, &n_names), nullptr);
    EXPECT_EQ(n_names, 0);
    EXPECT_EQ(aicore_rfdetr_context_class_names(nullptr, nullptr), nullptr);

    // Loading a nonexistent file must fail cleanly and report an error.
    aicore_rfdetr_ctx* ctx =
            aicore_rfdetr_load_opts("/nonexistent/rfdetr-model.gguf", opts);
    EXPECT_NE(ctx, nullptr);  // ctx is allocated; engine inside is null
    EXPECT_EQ(aicore_rfdetr_is_ready(ctx), 0);
    EXPECT_NE(aicore_rfdetr_last_error(ctx), nullptr);
    EXPECT_EQ(aicore_rfdetr_detect_path_json(ctx, "/nonexistent/image.png",
                                             0.5f, 300),
              nullptr);
    // Class names on a failed-load ctx: NULL (no class table to report).
    n_names = 99u;
    EXPECT_EQ(aicore_rfdetr_context_class_names(ctx, &n_names), nullptr);
    EXPECT_EQ(n_names, 0);
    aicore_rfdetr_free(ctx);
    aicore_rfdetr_options_free(opts);
}

TEST(RfdetrCApiContract, InvalidRgbAndMissingImageFail) {
    // RGB-buffer entry points reject invalid input.
    EXPECT_EQ(aicore_rfdetr_detect_rgb_json(nullptr, nullptr, 0, 0, 0.5f, 300),
              nullptr);

    uint8_t* rgb = nullptr;
    int32_t w = 0;
    int32_t h = 0;
    EXPECT_EQ(
            aicore_rfdetr_load_path_rgb("/nonexistent/image.png", &rgb, &w, &h),
            -1);
}

TEST(RfdetrCApiContract, CatalogPartitionsDetectionAndSegmentation) {
    // Model catalog contract.
    // 44 models = 11 variants (5 detection + 6 segmentation) x 4 quants
    // (f32, f16, q8_0, q4_K).
    EXPECT_EQ(aicore_rfdetr_model_count(), 44);
    EXPECT_EQ(aicore_rfdetr_detection_model_count(), 20);
    EXPECT_EQ(aicore_rfdetr_segmentation_model_count(), 24);
    EXPECT_EQ(aicore_rfdetr_detection_model_count() +
                      aicore_rfdetr_segmentation_model_count(),
              aicore_rfdetr_model_count());
    const aicore_rfdetr_model_entry* first = aicore_rfdetr_model_at(0);
    EXPECT_TRUE(first != nullptr && first->filename != nullptr &&
                first->download_url != nullptr);
    EXPECT_EQ(aicore_rfdetr_model_at(-1), nullptr);
    EXPECT_EQ(aicore_rfdetr_model_at(aicore_rfdetr_model_count()), nullptr);
    const aicore_rfdetr_model_entry* by_name =
            aicore_rfdetr_model_by_filename(first->filename);
    EXPECT_TRUE(by_name != nullptr &&
                std::strcmp(by_name->download_url, first->download_url) == 0);
    EXPECT_EQ(aicore_rfdetr_model_by_filename("nope.gguf"), nullptr);
    EXPECT_TRUE(aicore_rfdetr_model_download_base() != nullptr &&
                std::strstr(aicore_rfdetr_model_download_base(),
                            "RF-DETR-GGUF") != nullptr);

    // Role-tagged catalog accessors.
    const aicore_rfdetr_model_entry* det0 = aicore_rfdetr_detection_model_at(0);
    EXPECT_TRUE(det0 != nullptr && det0->segmentation_capable == 0);
    EXPECT_EQ(aicore_rfdetr_detection_model_at(20), nullptr);
    const aicore_rfdetr_model_entry* seg0 =
            aicore_rfdetr_segmentation_model_at(0);
    EXPECT_TRUE(seg0 != nullptr && seg0->segmentation_capable == 1);
    EXPECT_EQ(aicore_rfdetr_segmentation_model_at(24), nullptr);
}

TEST(RfdetrCApiContract, FailedLoadIntrospectionAndInfoRelease) {
    // Model-free introspection on a ctx with no loaded model.
    aicore_rfdetr_ctx* empty_ctx =
            aicore_rfdetr_load_opts("/nonexistent/rfdetr.gguf", nullptr);
    EXPECT_NE(empty_ctx, nullptr);
    EXPECT_EQ(aicore_rfdetr_is_ready(empty_ctx), 0);
    EXPECT_NE(aicore_rfdetr_context_variant(empty_ctx), nullptr);
    EXPECT_EQ(aicore_rfdetr_context_image_size(empty_ctx), 0);
    EXPECT_EQ(aicore_rfdetr_context_num_classes(empty_ctx), 0);
    EXPECT_EQ(aicore_rfdetr_context_has_segmentation(empty_ctx), 0);
    EXPECT_NE(aicore_rfdetr_context_device(empty_ctx), nullptr);
    // No model loaded -> no engine -> threads must read as 0.
    EXPECT_EQ(aicore_rfdetr_context_threads(empty_ctx), 0);
    // info_json returns NULL without a loaded model (no metadata to serialize);
    // free_string must accept NULL.
    char* info = aicore_rfdetr_info_json(empty_ctx);
    aicore_rfdetr_free_buffer(info);
    aicore_rfdetr_free(empty_ctx);
}

TEST(RfdetrCApiContract, CpuWarmupAndIdempotentShutdown) {
    // Device enumeration / warmup.
    EXPECT_EQ(aicore_rfdetr_warmup_backend("cpu"), 0);
    aicore_rfdetr_shutdown();
    aicore_rfdetr_shutdown();  // idempotent
}

TEST(RfdetrCApiContract, OwnedModelCacheDirectory) {
    char* dir = aicore_rfdetr_model_cache_dir();
    EXPECT_TRUE(dir != nullptr && std::strlen(dir) > 0);
    aicore_rfdetr_free_buffer(dir);
}
