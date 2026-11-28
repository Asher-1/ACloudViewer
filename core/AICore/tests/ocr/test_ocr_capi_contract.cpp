// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <aicore/ocr_capi.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <limits>

namespace {

bool errorContains(const char* needle) {
    const char* error = aicore_ocr_last_error();
    return error && std::strstr(error, needle);
}

class OcrCApiContract : public ::testing::Test {
protected:
    void SetUp() override {
        options.struct_size = sizeof(options);
        options.abi_version = AICORE_OCR_ABI_VERSION;
        options.det_model_path = "/does/not/exist-det.gguf";
        options.rec_model_path = "/does/not/exist-rec.gguf";
    }

    aicore_ocr_options options{};
};

}  // namespace

TEST(OcrPresetContract, PublishedRecipesPreserveDetectorAndThresholds) {
    const auto* tiny =
            aicore_ocr_preset_by_rec_filename("ppocrv6-tiny-rec-f32.gguf");
    const auto* small =
            aicore_ocr_preset_by_rec_filename("ppocrv6-small-rec-f32.gguf");
    const auto* japan =
            aicore_ocr_preset_by_rec_filename("ppocrv3-japan-rec-f32.gguf");
    const auto* sourceV4 =
            aicore_ocr_preset_by_rec_filename("ppocrv4-source-rec-f32.gguf");
    ASSERT_NE(japan, nullptr);
    ASSERT_NE(sourceV4, nullptr);
    EXPECT_STREQ(japan->det_stem, sourceV4->det_stem);
    EXPECT_EQ(std::strncmp(japan->display_name, "japan_PP-OCRv3", 14), 0);
    ASSERT_NE(tiny, nullptr);
    ASSERT_NE(small, nullptr);
    EXPECT_EQ(tiny->det_thresh, .2f);
    EXPECT_EQ(tiny->det_box_thresh, .4f);
    EXPECT_EQ(small->det_box_thresh, .45f);
    EXPECT_EQ(tiny->det_max_candidates, 3000);
    EXPECT_EQ(tiny->cls_thresh, .9f);
    EXPECT_EQ(tiny->rec_batch, 6);
    EXPECT_EQ(aicore_ocr_preset_at(-1), nullptr);
    EXPECT_EQ(aicore_ocr_preset_at(aicore_ocr_preset_count()), nullptr);
    EXPECT_EQ(aicore_ocr_preset_by_rec_filename("unknown.gguf"), nullptr);
    // Published recipes must also resolve as real model download assets.
    EXPECT_NE(aicore_ocr_preset_by_rec_filename("ppocrv6-tiny-rec-f32.gguf"),
              nullptr);
    EXPECT_NE(aicore_ocr_model_by_filename("ppocrv6-tiny-rec-f32.gguf"),
              nullptr);
}

TEST_F(OcrCApiContract, OriginalOptionsSizeReachesModelLoading) {
    // The original ABI ended before device; loading owns this failure.
    options.struct_size = offsetof(aicore_ocr_options, device);
    auto* ctx = aicore_ocr_context_create(&options);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_TRUE(errorContains("det model load failed"));
    aicore_ocr_context_destroy(ctx);
}

TEST_F(OcrCApiContract, LegacyOptionsDoNotReadTrailingClassifierFields) {
    options.struct_size = offsetof(aicore_ocr_options, cls_model_path);
    options.cls_thresh = std::numeric_limits<float>::quiet_NaN();
    auto* ctx = aicore_ocr_context_create(&options);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_TRUE(errorContains("det model load failed"));
    aicore_ocr_context_destroy(ctx);

    options.struct_size = sizeof(options);
    ctx = aicore_ocr_context_create(&options);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_TRUE(errorContains("invalid classifier"));
    aicore_ocr_context_destroy(ctx);
}

TEST_F(OcrCApiContract, TruncatedOptionsAreRejected) {
    options.struct_size = offsetof(aicore_ocr_options, rec_max_batch);
    auto* ctx = aicore_ocr_context_create(&options);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_TRUE(errorContains("struct_size"));
    aicore_ocr_context_destroy(ctx);
}

TEST_F(OcrCApiContract, FutureAbiVersionIsRejected) {
    options.abi_version = AICORE_OCR_ABI_VERSION + 1;
    auto* ctx = aicore_ocr_context_create(&options);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_TRUE(errorContains("abi version mismatch"));
    aicore_ocr_context_destroy(ctx);
}

TEST_F(OcrCApiContract, UnsupportedKieOptionIsRejected) {
    options.kie_model_path = "/does/not/exist-kie.gguf";
    auto* ctx = aicore_ocr_context_create(&options);
    EXPECT_EQ(ctx, nullptr);
    EXPECT_TRUE(errorContains("KIE models are not supported"));
    aicore_ocr_context_destroy(ctx);
}

TEST(OcrNullContract, RecognitionAndResultAccessorsReturnEmptyDefaults) {
    EXPECT_EQ(aicore_ocr_recognize_regions(nullptr, nullptr, nullptr, 0),
              nullptr);
    EXPECT_EQ(aicore_ocr_context_ready(nullptr), 0);
    EXPECT_STREQ(aicore_ocr_context_device(nullptr), "");
    EXPECT_EQ(aicore_ocr_label_count(nullptr), 0);
    EXPECT_EQ(aicore_ocr_label_at(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_ocr_box_count(nullptr), 0);
    EXPECT_EQ(aicore_ocr_box_at(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_ocr_region_count(nullptr), 0);
    EXPECT_EQ(aicore_ocr_region_at(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_ocr_relation_count(nullptr), 0);
    EXPECT_EQ(aicore_ocr_relation_at(nullptr, 0), nullptr);
    EXPECT_EQ(aicore_ocr_orientation_at(nullptr, 0), nullptr);
}

TEST(OcrNullContract, TimingValidityIsEmptyAndResultReleaseIsSafe) {
    aicore_ocr_result* result = nullptr;
    const auto timings = aicore_ocr_timings(result);
    const auto post = aicore_ocr_post_timings(result);
    EXPECT_EQ(timings.valid_fields, 0);
    EXPECT_EQ(post.valid_fields, 0);
    aicore_ocr_result_release(result);
}

TEST(OcrNullContract, RuntimeShutdownIsIdempotent) {
    aicore_ocr_runtime_shutdown();
    aicore_ocr_runtime_shutdown();
}
