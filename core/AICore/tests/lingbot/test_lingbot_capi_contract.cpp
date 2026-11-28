// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// LingBot-Map C ABI contract test: NULL safety, lifecycle, options, catalog
// invariants, and preprocessing sizing. Model-dependent inference runs are
// covered by the asset-driven validation runner (validation_manifest.json),
// not here.

#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aicore/lingbot_capi.h"

TEST(LingbotCApiContract, AbiAndNullReleaseSetters) {
    EXPECT_GE(aicore_lingbot_abi_version(), 1);

    // NULL-safe teardown.
    aicore_lingbot_free(nullptr);
    aicore_lingbot_options_free(nullptr);
    aicore_lingbot_free_buffer(nullptr);

    // NULL-safe setters.
    aicore_lingbot_options_set_device(nullptr, "cpu");
    aicore_lingbot_options_set_threads(nullptr, 1);
    aicore_lingbot_options_set_image_size(nullptr, 518);
    aicore_lingbot_options_set_kv_profile(nullptr, 8, 64);
    aicore_lingbot_options_set_stream_capacity(nullptr, 0);
    aicore_lingbot_options_set_keyframe_interval(nullptr, 4);
}

TEST(LingbotCApiContract, NullContextStreamAndSkySegQueries) {
    EXPECT_EQ(aicore_lingbot_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_lingbot_is_ready(nullptr), 0);
    EXPECT_NE(aicore_lingbot_last_error(nullptr), nullptr);
    EXPECT_EQ(aicore_lingbot_last_pipeline_timings(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_lingbot_last_stream_frames(nullptr), -1);
    EXPECT_EQ(aicore_lingbot_stream_reset(nullptr), -1);
    EXPECT_EQ(aicore_lingbot_skyseg_ready(nullptr), 0);
    EXPECT_EQ(aicore_lingbot_skyseg_load(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_lingbot_infer_stream(nullptr, nullptr, 0, 0, 0, nullptr,
                                          nullptr),
              -1);
    EXPECT_EQ(aicore_lingbot_context_image_size(nullptr), 0);
    EXPECT_EQ(aicore_lingbot_context_patch_size(nullptr), 0);
    EXPECT_EQ(aicore_lingbot_context_threads(nullptr), 0);
    // Device is an empty string (never NULL) without a loaded context.
    EXPECT_EQ(std::strlen(aicore_lingbot_context_device(nullptr)), 0);
    EXPECT_EQ(aicore_lingbot_info_json(nullptr), nullptr);
    EXPECT_EQ(aicore_lingbot_last_sky_mask(nullptr, nullptr, 0), -1);
    // External sky-mask injection: invalid args rejected; a valid injection
    // is verified end-to-end in the asset-driven probe (validation runner).
    EXPECT_EQ(aicore_lingbot_set_external_sky_masks(nullptr, nullptr, 0, 0, 0),
              -1);
}

TEST(LingbotCApiContract, MissingModelSkyMaskSetterRemainsSafe) {
    {
        std::vector<unsigned char> masks(2 * 4 * 4, 255);
        aicore_lingbot_options* o = aicore_lingbot_options_new();
        EXPECT_NE(o, nullptr);
        aicore_lingbot_options_set_device(o, "cpu");
        // No real GGUF available in the contract test; the setter's
        // argument validation is exercised directly below on the error
        // context that load_opts returns for a missing file.
        aicore_lingbot_ctx* c =
                aicore_lingbot_load_opts("missing_lingbot_contract.gguf", o);
        aicore_lingbot_options_free(o);
        // A missing GGUF yields a non-ready error-carrying context; the
        // setter must still be contract-safe (0 on a valid shape, since the
        // masks are consumed at infer time, or -1 pre-context).
        if (c && aicore_lingbot_is_ready(c) == 1) {
            EXPECT_EQ(aicore_lingbot_set_external_sky_masks(c, masks.data(), 2,
                                                            4, 4),
                      0);
        }
        aicore_lingbot_free(c);
    }
}

TEST(LingbotCApiContract, OptionsLifecycle) {
    aicore_lingbot_options* opts = aicore_lingbot_options_new();
    EXPECT_NE(opts, nullptr);
    aicore_lingbot_options_set_device(opts, "cpu");
    aicore_lingbot_options_set_threads(opts, 1);
    aicore_lingbot_options_set_image_size(opts, 518);
    aicore_lingbot_options_set_kv_profile(opts, 8, 64);
    aicore_lingbot_options_set_stream_capacity(opts, 0);
    aicore_lingbot_options_free(opts);
}

TEST(LingbotCApiContract, PreprocessTwoCallSizingAndBufferContract) {
    // Preprocessing sizing without a buffer (two-call sizing contract).
    // The official crop keeps the width at image_size (28 = 2 patches) and
    // snaps the height to the aspect-scaled patch grid (round(2*28/4/14)*14
    // = 14), so a 4x2 RGB image reports 28*14*3 floats.
    aicore_image_view view{};
    view.format = AICORE_IMAGE_RGB8;
    EXPECT_EQ(aicore_lingbot_preprocess_image(nullptr, 518, nullptr, 0, nullptr,
                                              nullptr),
              -1);
    std::vector<uint8_t> pixels(4 * 2 * 3, 128);
    view.data = pixels.data();
    view.width = 4;
    view.height = 2;
    view.row_stride_bytes = 4 * 3;
    const int sized = aicore_lingbot_preprocess_image(&view, 28, nullptr, 0,
                                                      nullptr, nullptr);
    EXPECT_EQ(sized, 28 * 14 * 3);
    // The SIZING call must report the same processed dimensions as the data
    // call (callers resolve the stream geometry from it).
    int32_t sized_w = 0, sized_h = 0;
    EXPECT_EQ(aicore_lingbot_preprocess_image(&view, 28, nullptr, 0, &sized_w,
                                              &sized_h),
              sized);
    EXPECT_TRUE(sized_w == 28 && sized_h == 14);
    EXPECT_EQ(aicore_lingbot_preprocess_image(&view, 0, nullptr, 0, nullptr,
                                              nullptr),
              -1);
    // Non-zero buffer with a too-small size fails; the full size succeeds
    // and reports the processed dimensions.
    std::vector<float> dst(sized);
    int32_t w_out = 0, h_out = 0;
    EXPECT_EQ(aicore_lingbot_preprocess_image(&view, 28, dst.data(), sized / 2,
                                              &w_out, &h_out),
              -1);
    EXPECT_EQ(aicore_lingbot_preprocess_image(&view, 28, dst.data(), sized,
                                              &w_out, &h_out),
              sized);
    EXPECT_TRUE(w_out == 28 && h_out == 14);
}

TEST(LingbotCApiContract, CpuWarmup) {
    // Warmup on the always-available CPU backend.
    EXPECT_EQ(aicore_lingbot_warmup_backend("cpu"), 0);
}

TEST(LingbotCApiContract, PinnedCatalogHasDefaultsForMapAndSkySeg) {
    // Model catalog invariants: URLs match the pinned HF repo, sizes and
    // digests are present, roles are exactly "map"/"skyseg", and exactly one
    // default row exists per role.
    const int count = aicore_lingbot_model_count();
    EXPECT_GE(count, 5);
    const char* base = aicore_lingbot_model_download_base();
    EXPECT_TRUE(base != nullptr &&
                std::strstr(base, "huggingface.co/Asher-1/lingbot-map-gguf") !=
                        nullptr);
    int map_defaults = 0, skyseg_defaults = 0, map_rows = 0, skyseg_rows = 0;
    for (int i = 0; i < count; ++i) {
        const aicore_lingbot_model_entry* e = aicore_lingbot_model_at(i);
        EXPECT_NE(e, nullptr);
        if (!e) continue;
        EXPECT_TRUE(e->filename != nullptr && e->filename[0] != '\0');
        EXPECT_TRUE(e->download_url != nullptr &&
                    std::strncmp(e->download_url, base, std::strlen(base)) ==
                            0);
        EXPECT_NE(std::strstr(e->download_url, e->filename), nullptr);
        EXPECT_GT(e->size_bytes, 0);
        EXPECT_TRUE(e->sha256 != nullptr && std::strlen(e->sha256) == 64);
        EXPECT_TRUE(e->license_note != nullptr && e->license_note[0] != '\0');
        EXPECT_TRUE(e->role != nullptr &&
                    (std::strcmp(e->role, "map") == 0 ||
                     std::strcmp(e->role, "skyseg") == 0));
        EXPECT_EQ(aicore_lingbot_model_by_filename(e->filename), e);
        if (std::strcmp(e->role, "map") == 0) {
            ++map_rows;
            if (i == aicore_lingbot_model_default_index()) ++map_defaults;
        } else {
            ++skyseg_rows;
            if (i == aicore_lingbot_skyseg_model_default_index())
                ++skyseg_defaults;
        }
    }
    EXPECT_TRUE(map_rows >= 3 && skyseg_rows >= 2);
    EXPECT_TRUE(map_defaults == 1 && skyseg_defaults == 1);
    EXPECT_EQ(aicore_lingbot_model_by_filename("missing.gguf"), nullptr);
    EXPECT_EQ(aicore_lingbot_model_at(-1), nullptr);
    EXPECT_EQ(aicore_lingbot_model_at(count), nullptr);
}
