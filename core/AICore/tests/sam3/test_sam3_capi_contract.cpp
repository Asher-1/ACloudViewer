// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// SAM2 / SAM2.1 / SAM3 C API contract test — fast, no GGUF assets required.
// Covers ABI, options lifecycle, null-safe error paths, geometry accessors,
// tracker plumbing, timings and the published model catalog.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/sam3_capi.h"

TEST(Sam3CApiContract, AbiAndNullReleaseQueries) {
    EXPECT_GE(aicore_sam3_abi_version(), 1);

    // Null-safe teardown / lifecycle.
    aicore_sam3_free(nullptr);
    aicore_sam3_options_free(nullptr);
    aicore_sam3_free_buffer(nullptr);
    aicore_sam3_seg_result_free(nullptr);
    aicore_sam3_tracker_free(nullptr);
    aicore_sam3_tracker_reset(nullptr);

    EXPECT_EQ(aicore_sam3_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(std::strcmp(aicore_sam3_last_load_error(), "empty model path"),
              0);
    EXPECT_EQ(aicore_sam3_is_ready(nullptr), 0);
    EXPECT_NE(aicore_sam3_last_error(nullptr), nullptr);
    EXPECT_EQ(aicore_sam3_context_model_type(nullptr), -1);
    EXPECT_EQ(aicore_sam3_context_visual_only(nullptr), 0);
    EXPECT_NE(aicore_sam3_context_backend_name(nullptr), nullptr);
    EXPECT_EQ(std::strcmp(aicore_sam3_context_backend_name(nullptr), "none"),
              0);
    EXPECT_EQ(aicore_sam3_context_threads(nullptr), 0);
    EXPECT_EQ(aicore_sam3_has_encoded_image(nullptr), 0);
    EXPECT_EQ(aicore_sam3_tracker_frame_index(nullptr), -1);
    EXPECT_NE(aicore_sam3_tracker_last_error(nullptr), nullptr);
    EXPECT_EQ(aicore_sam3_set_score_threshold(nullptr, 0.5f), -1);
}

TEST(Sam3CApiContract, OptionsLifecycleAndNullSetters) {
    // Options lifecycle + NULL no-ops.
    aicore_sam3_options* opts = aicore_sam3_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_sam3_options_set_device(opts, "cpu");
    aicore_sam3_options_set_threads(opts, 1);
    aicore_sam3_options_set_encode_img_size(opts, 1024);
    aicore_sam3_options_set_score_threshold(opts, 0.5f);
    aicore_sam3_options_set_nms_threshold(opts, 0.1f);
    aicore_sam3_options_set_assoc_iou_threshold(opts, 0.1f);
    aicore_sam3_options_set_hotstart_delay(opts, 15);
    aicore_sam3_options_set_max_keep_alive(opts, 30);
    aicore_sam3_options_set_recondition_every(opts, 16);
    aicore_sam3_options_set_fill_hole_area(opts, 16);
    aicore_sam3_options_set_device(nullptr, "cpu");
    aicore_sam3_options_set_threads(nullptr, 1);
    aicore_sam3_options_set_encode_img_size(nullptr, 1024);
    aicore_sam3_options_set_score_threshold(nullptr, 0.5f);
    aicore_sam3_options_set_nms_threshold(nullptr, 0.1f);
    aicore_sam3_options_set_assoc_iou_threshold(nullptr, 0.1f);
    aicore_sam3_options_set_hotstart_delay(nullptr, 15);
    aicore_sam3_options_set_max_keep_alive(nullptr, 30);
    aicore_sam3_options_set_recondition_every(nullptr, 16);
    aicore_sam3_options_set_fill_hole_area(nullptr, 16);
    aicore_sam3_options_free(opts);
}

TEST(Sam3CApiContract, MissingModelReportsLoadError) {
    // Loading nonexistent model files must fail cleanly with a null ctx and
    // no crash.
    aicore_sam3_ctx* ctx =
            aicore_sam3_load_opts("/nonexistent/sam3-f16.gguf", nullptr);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_NE(aicore_sam3_last_load_error()[0], '\0');
    aicore_sam3_free(ctx);
}

TEST(Sam3CApiContract, NullImageEncodingAndSegmentationFail) {
    // Image/geometry entry points with a null ctx: fail without crashing.
    static const uint8_t kRgb[4 * 3] = {0};
    EXPECT_EQ(aicore_sam3_encode_rgb(nullptr, kRgb, 2, 2, sizeof(kRgb), 0), -1);
    /* F-01 batch A: image_view entry is NULL-safe and rejects non-RGB8
     * formats without touching model state. */
    EXPECT_EQ(aicore_sam3_encode_image_view(nullptr, nullptr, 0), -1);
    aicore_sam3_pcs_prompt pcs{};
    pcs.text = "cat";
    EXPECT_EQ(aicore_sam3_segment_pcs_rgb(nullptr, &pcs, kRgb, 2, 2,
                                          sizeof(kRgb)),
              nullptr);
    aicore_sam3_pvs_prompt pvs{};
    pvs.use_box = 1;
    EXPECT_EQ(aicore_sam3_segment_pvs_rgb(nullptr, &pvs, kRgb, 2, 2,
                                          sizeof(kRgb)),
              nullptr);
}

TEST(Sam3CApiContract, NullSegmentationResultDefaults) {
    // Seg result accessors on a null result return defaults.
    EXPECT_EQ(aicore_sam3_seg_det_count(nullptr), 0);
    const aicore_sam3_box box = aicore_sam3_seg_det_box_at(nullptr, 0);
    EXPECT_TRUE(box.x0 == 0.0f && box.x1 == 0.0f);
    EXPECT_EQ(aicore_sam3_seg_det_score_at(nullptr, 0), 0.0f);
    EXPECT_EQ(aicore_sam3_seg_det_iou_at(nullptr, 0), 0.0f);
    EXPECT_EQ(aicore_sam3_seg_det_instance_id_at(nullptr, 0), -1);
    const aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(nullptr, 0);
    EXPECT_TRUE(mask.data == nullptr && mask.width == 0);
}

TEST(Sam3CApiContract, NullQuantizationFails) {
    // Quantize: NULL arguments must fail without crashing.
    EXPECT_EQ(aicore_sam3_quantize_gguf(nullptr, nullptr, nullptr), -1);
}

TEST(Sam3CApiContract, InvalidBenchmarkAndProfileFail) {
    aicore_sam3_ctx* ctx = nullptr;
    // Benchmark: NULL ctx / out pointer or bad dims must fail without
    // crashing.
    aicore_sam3_timings bench{};
    EXPECT_EQ(aicore_sam3_benchmark(nullptr, 64, 64, 1, 1, &bench), -1);
    EXPECT_EQ(aicore_sam3_benchmark(ctx, 64, 64, 1, 1, nullptr), -1);
    EXPECT_EQ(aicore_sam3_benchmark(ctx, 0, 64, 1, 1, &bench), -1);
    EXPECT_EQ(aicore_sam3_benchmark(ctx, 64, 0, 1, 1, &bench), -1);

    // Profile: NULL ctx must fail; a loaded-model probe is allowed to
    // report 0 entries on backends without the profile sub-graphs but must
    // not crash.
    aicore_sam3_profile_entry prof[8];
    int n_prof = -1;
    EXPECT_EQ(aicore_sam3_profile_encoder(nullptr, 1, 1, prof, 8, &n_prof), -1);
}

TEST(Sam3CApiContract, NullTrackerPlumbingFails) {
    static const uint8_t kRgb[4 * 3] = {0};
    aicore_sam3_pvs_prompt pvs{};
    pvs.use_box = 1;
    const aicore_sam3_plane_view mask = aicore_sam3_seg_mask_at(nullptr, 0);
    // Tracker plumbing on a null ctx/tracker fails cleanly.
    EXPECT_EQ(aicore_sam3_tracker_create(nullptr), nullptr);
    aicore_sam3_tracker_set_text_prompt(nullptr, "cat");  // NULL-safe no-op
    EXPECT_EQ(aicore_sam3_track_frame(nullptr, kRgb, 2, 2, sizeof(kRgb)),
              nullptr);
    EXPECT_EQ(aicore_sam3_propagate_frame(nullptr, kRgb, 2, 2, sizeof(kRgb)),
              nullptr);
    EXPECT_EQ(aicore_sam3_tracker_add_instance(nullptr, &pvs), -1);
    EXPECT_EQ(aicore_sam3_tracker_add_instance_from_mask(nullptr, &mask), -1);
    EXPECT_EQ(aicore_sam3_refine_instance(nullptr, 0, nullptr, 0, nullptr, 0),
              -1);
    // PVS on the tracker's encoded frame: NULL-safe, and a null prompt fails
    // cleanly.
    EXPECT_EQ(aicore_sam3_tracker_segment_pvs(nullptr, &pvs), nullptr);
    EXPECT_EQ(aicore_sam3_tracker_segment_pvs(nullptr, nullptr), nullptr);
}

TEST(Sam3CApiContract, NullTimingsFail) {
    aicore_sam3_ctx* ctx = nullptr;
    // Timings: null ctx or null out pointer -> -1; valid pointer stays
    // untouched until a real inference runs.
    aicore_sam3_timings timings{};
    EXPECT_EQ(aicore_sam3_last_timings(nullptr, &timings), -1);
    EXPECT_EQ(aicore_sam3_last_timings(ctx, nullptr), -1);
    EXPECT_EQ(aicore_sam3_tracker_last_timings(nullptr, &timings), -1);
    EXPECT_EQ(aicore_sam3_tracker_last_timings(nullptr, nullptr), -1);
}

TEST(Sam3CApiContract, CatalogUrlsRemainStableAcrossQueries) {
    // Published model catalog.
    const int count = aicore_sam3_model_count();
    EXPECT_GT(count, 0);
    EXPECT_EQ(aicore_sam3_model_at(-1), nullptr);
    EXPECT_EQ(aicore_sam3_model_at(count), nullptr);
    const aicore_sam3_model_entry* e0 = aicore_sam3_model_at(0);
    EXPECT_NE(e0, nullptr);
    EXPECT_TRUE(e0->filename != nullptr && e0->filename[0] != '\0');
    EXPECT_NE(e0->download_url, nullptr);
    EXPECT_EQ(std::strstr(e0->download_url, "http"), e0->download_url);
    EXPECT_EQ(aicore_sam3_model_by_filename(e0->filename), e0);
    EXPECT_EQ(aicore_sam3_model_by_filename("no-such-model.gguf"), nullptr);
    EXPECT_EQ(aicore_sam3_model_by_filename(nullptr), nullptr);
    const char* base = aicore_sam3_model_download_base();
    EXPECT_TRUE(base != nullptr && base[0] != '\0');
    std::vector<std::string> catalogUrls;
    catalogUrls.reserve(static_cast<size_t>(count));
    for (int i = 0; i < count; ++i) {
        const aicore_sam3_model_entry* entry = aicore_sam3_model_at(i);
        EXPECT_TRUE(entry != nullptr && entry->download_url != nullptr);
        if (entry) {
            const std::string expected =
                    std::string(base) + std::string(entry->filename);
            EXPECT_EQ(expected, entry->download_url);
            catalogUrls.emplace_back(entry->download_url);
        }
    }
    for (int i = 0; i < count; ++i) {
        const aicore_sam3_model_entry* entry = aicore_sam3_model_at(i);
        EXPECT_TRUE(entry != nullptr &&
                    catalogUrls[static_cast<size_t>(i)] == entry->download_url);
    }
}

TEST(Sam3CApiContract, WarmupAndIdempotentShutdown) {
    // Process-wide helpers: must not crash on an uninitialized process.
    aicore_sam3_warmup_backend("cpu");
    aicore_sam3_shutdown();
    aicore_sam3_shutdown();  // idempotent
}
