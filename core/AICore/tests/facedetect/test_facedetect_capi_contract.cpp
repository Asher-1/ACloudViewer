// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// FaceDetect C API contract tests require no GGUF assets. Model inference is
// exercised by the asset-dependent C API model tests.

#include <gtest/gtest.h>

#include <cstring>

#include "aicore/facedetect_capi.h"

TEST(FacedetectCApiContract, AbiAndNullContextGuards) {
    EXPECT_GE(aicore_facedetect_abi_version(), 2);
    aicore_facedetect_free(nullptr);
    aicore_facedetect_options_free(nullptr);
    aicore_facedetect_free_buffer(nullptr);
    aicore_facedetect_free_buffer(nullptr);

    EXPECT_EQ(aicore_facedetect_load_opts(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_facedetect_is_ready(nullptr), 0);
    EXPECT_EQ(aicore_facedetect_last_error(nullptr), nullptr);
    EXPECT_EQ(aicore_facedetect_detect_image(nullptr, nullptr), -1);
    EXPECT_EQ(aicore_facedetect_analyze_image(nullptr, nullptr, 0.0f), -1);
    EXPECT_EQ(aicore_facedetect_analysis_count(nullptr), 0);
    EXPECT_EQ(aicore_facedetect_analysis_at(nullptr, 0, nullptr), -1);
    int32_t embedding_dim = 17;
    EXPECT_TRUE(aicore_facedetect_analysis_embedding(
                        nullptr, 0, &embedding_dim) == nullptr &&
                embedding_dim == 0);
    EXPECT_EQ(aicore_facedetect_dense_landmarks_image(nullptr, nullptr, nullptr,
                                                      0.0f),
              -1);
    EXPECT_EQ(aicore_facedetect_dense_face_count(nullptr), 0);
    EXPECT_EQ(aicore_facedetect_dense_detection_at(nullptr, 0, nullptr), -1);
    EXPECT_EQ(aicore_facedetect_dense_point_count(nullptr, 0, 0), 0);
    EXPECT_EQ(aicore_facedetect_dense_point_at(nullptr, 0, 0, 0, nullptr), -1);
    EXPECT_EQ(aicore_facedetect_verify_images(nullptr, nullptr, nullptr,
                                              nullptr, nullptr),
              -1);
    EXPECT_EQ(aicore_facedetect_last_pipeline_timings(nullptr, nullptr), -1);
}

TEST(FacedetectCApiContract, OptionsAndMissingModelExposeErrorContext) {
    aicore_facedetect_options* opts = aicore_facedetect_options_new();
    ASSERT_NE(opts, nullptr);
    aicore_facedetect_options_set_device(opts, "cpu");
    aicore_facedetect_options_set_threads(opts, 1);
    aicore_facedetect_options_set_device(nullptr, "cpu");
    aicore_facedetect_options_set_threads(nullptr, 1);

    aicore_facedetect_ctx* ctx =
            aicore_facedetect_load_opts("/nonexistent/fd.gguf", opts);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(aicore_facedetect_is_ready(ctx), 0);
    EXPECT_NE(aicore_facedetect_last_error(ctx), nullptr);
    EXPECT_NE(aicore_facedetect_info_json(ctx), nullptr);
    aicore_facedetect_free(ctx);
    aicore_facedetect_options_free(opts);
}

TEST(FacedetectCApiContract, MissingImageLoadClearsOutputs) {
    uint8_t* rgb = nullptr;
    int32_t width = 0;
    int32_t height = 0;
    EXPECT_EQ(aicore_facedetect_load_path_rgb("/nonexistent/x.png", &rgb,
                                              &width, &height),
              -1);
    EXPECT_TRUE(rgb == nullptr && width == 0 && height == 0);
}

TEST(FacedetectCApiContract, CatalogEntriesAndRoleViewsAreConsistent) {
    const int total = aicore_facedetect_model_count();
    ASSERT_GT(total, 0);
    EXPECT_EQ(aicore_facedetect_model_at(-1), nullptr);
    EXPECT_EQ(aicore_facedetect_model_at(total), nullptr);
    for (int i = 0; i < total; ++i) {
        const aicore_facedetect_model_entry* entry =
                aicore_facedetect_model_at(i);
        ASSERT_NE(entry, nullptr);
        EXPECT_NE(entry->filename, nullptr);
        EXPECT_NE(entry->download_url, nullptr);
        EXPECT_NE(entry->display_name, nullptr);
        EXPECT_NE(entry->license_note, nullptr);
    }
    const int detector_count = aicore_facedetect_detector_model_count();
    const int landmark_count = aicore_facedetect_landmark_model_count();
    EXPECT_GT(detector_count, 0);
    EXPECT_GE(landmark_count, 0);
    EXPECT_EQ(detector_count + landmark_count, total);
    EXPECT_NE(aicore_facedetect_detector_model_at(0), nullptr);
    EXPECT_EQ(aicore_facedetect_detector_model_at(detector_count), nullptr);
    if (landmark_count > 0) {
        EXPECT_NE(aicore_facedetect_landmark_model_at(0), nullptr);
        EXPECT_EQ(aicore_facedetect_landmark_model_at(landmark_count), nullptr);
    }

    const aicore_facedetect_model_entry* first = aicore_facedetect_model_at(0);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(aicore_facedetect_model_by_filename(first->filename), first);
    EXPECT_EQ(aicore_facedetect_model_by_filename("nope.gguf"), nullptr);
    ASSERT_NE(aicore_facedetect_model_download_base(), nullptr);
    EXPECT_NE(
            std::strstr(aicore_facedetect_model_download_base(), "qFaceDetect"),
            nullptr);
}

TEST(FacedetectCApiContract, CompatibilityWrappersRejectNullContext) {
    EXPECT_EQ(aicore_facedetect_analyze_rgb_json(nullptr, nullptr, 0, 0, 0.0f),
              nullptr);
    EXPECT_EQ(aicore_facedetect_dense_landmarks_rgb_json(nullptr, nullptr,
                                                         nullptr, 0, 0, 0.0f),
              nullptr);
    float verification_distance = 0.0f;
    int verified = 0;
    EXPECT_EQ(
            aicore_facedetect_verify_paths(nullptr, nullptr, nullptr, -1.0f, 0,
                                           &verification_distance, &verified),
            -1);
}

TEST(FacedetectCApiContract, CacheWarmupAndShutdownLifecycle) {
    char* directory = aicore_facedetect_model_cache_dir();
    ASSERT_NE(directory, nullptr);
    EXPECT_GT(std::strlen(directory), 0);
    aicore_facedetect_free_buffer(directory);
    EXPECT_EQ(aicore_facedetect_warmup_backend("cpu"), 0);
    aicore_facedetect_shutdown();
    aicore_facedetect_shutdown();
}
