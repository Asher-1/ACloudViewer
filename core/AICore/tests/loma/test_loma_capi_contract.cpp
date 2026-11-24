// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>

#include "aicore/aicore.h"

TEST(LomaCApiContract, AbiAndRoleVariantCatalog) {
    EXPECT_EQ(aicore_loma_abi_version(), 6);
    EXPECT_EQ(aicore_loma_model_count(), 22);
    /* Catalog metadata: every role has a default entry, an unknown role is
     * rejected, and the download base is a non-empty URL string. */
    EXPECT_GE(aicore_loma_model_default_index(AICORE_LOMA_MODEL_ROLE_DETECTOR),
              0);
    EXPECT_GE(aicore_loma_model_default_index(AICORE_LOMA_MODEL_ROLE_MATCHER),
              0);
    EXPECT_EQ(aicore_loma_model_default_index((aicore_loma_model_role)999), -1);
    EXPECT_TRUE(aicore_loma_model_download_base() != nullptr &&
                std::strlen(aicore_loma_model_download_base()) > 0);
    for (int role = AICORE_LOMA_MODEL_ROLE_DETECTOR;
         role <= AICORE_LOMA_MODEL_ROLE_MATCHER; ++role) {
        const aicore_loma_model_entry* model = aicore_loma_model_by_role(
                static_cast<aicore_loma_model_role>(role));
        EXPECT_NE(model, nullptr);
        EXPECT_TRUE(model->filename != nullptr && model->filename[0] != '\0');
        EXPECT_TRUE(model->download_url != nullptr &&
                    model->download_url[0] != '\0');
        EXPECT_EQ(model->role, role);
    }
    for (int variant = AICORE_LOMA_MODEL_VARIANT_DAD;
         variant <= AICORE_LOMA_MODEL_VARIANT_MATCHER_G; ++variant) {
        const aicore_loma_model_entry* model = aicore_loma_model_by_variant(
                static_cast<aicore_loma_model_variant>(variant));
        EXPECT_NE(model, nullptr);
        EXPECT_EQ(model->variant, variant);
    }
}

TEST(LomaCApiContract, OwnedModelCacheDirectory) {
    char* model_cache = aicore_loma_model_cache_dir();
    EXPECT_NE(model_cache, nullptr);
    std::free(model_cache);
}

TEST(LomaCApiContract, DetectorNullContract) {
    aicore_loma_detector_free(nullptr);
    aicore_loma_detector_options_free(nullptr);
    aicore_loma_detected_features empty{};
    aicore_loma_detected_features_free(&empty);
    EXPECT_EQ(aicore_loma_detector_load(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_loma_detector_is_ready(nullptr), 0);
    EXPECT_NE(aicore_loma_detector_last_error(nullptr), nullptr);
    EXPECT_NE(aicore_loma_detector_run(nullptr, nullptr, nullptr), 0);
    /* F-01 batch A: image_view entries are NULL-safe and reject non-RGB8
     * views before touching any model state. */
    EXPECT_NE(aicore_loma_detector_run_image_view(nullptr, nullptr, nullptr),
              0);
    aicore_pipeline_timings timings{};
    EXPECT_NE(aicore_loma_detector_last_pipeline_timings(nullptr, &timings), 0);
}

TEST(LomaCApiContract, DescriptorNullContract) {
    EXPECT_NE(aicore_loma_descriptor_run_image_view(nullptr, nullptr, nullptr,
                                                    0, 0, 0, nullptr),
              0);
    aicore_pipeline_timings timings{};
    EXPECT_NE(aicore_loma_descriptor_last_pipeline_timings(nullptr, &timings),
              0);
    aicore_loma_descriptor_free(nullptr);
    aicore_loma_descriptor_options_free(nullptr);
    aicore_loma_described_features empty_descriptors{};
    aicore_loma_described_features_free(&empty_descriptors);
    EXPECT_EQ(aicore_loma_descriptor_load(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_loma_descriptor_is_ready(nullptr), 0);
    EXPECT_NE(aicore_loma_descriptor_last_error(nullptr), nullptr);
    EXPECT_NE(aicore_loma_descriptor_run(nullptr, nullptr, nullptr, 0, 0, 0,
                                         nullptr),
              0);
}

TEST(LomaCApiContract, MatcherNullContract) {
    aicore_loma_matcher_free(nullptr);
    aicore_loma_free_matches(nullptr);
    aicore_loma_matcher_options_free(nullptr);
    EXPECT_EQ(aicore_loma_matcher_load(nullptr, nullptr), nullptr);
    EXPECT_EQ(aicore_loma_matcher_is_ready(nullptr), 0);
    EXPECT_NE(aicore_loma_matcher_last_error(nullptr), nullptr);
    EXPECT_NE(aicore_loma_matcher_run(nullptr, nullptr, nullptr, nullptr,
                                      nullptr),
              0);
    aicore_pipeline_timings timings{};
    /* F-10 (matcher shape): the matcher timing query is NULL-safe too. */
    EXPECT_NE(aicore_loma_matcher_last_pipeline_timings(nullptr, &timings), 0);
}

TEST(LomaCApiContract, MatcherOptionsLifecycle) {
    aicore_loma_matcher_options* options = aicore_loma_matcher_options_new();
    EXPECT_NE(options, nullptr);
    aicore_loma_matcher_options_set_device(options, "cpu");
    aicore_loma_matcher_options_set_threads(options, 1);
    aicore_loma_matcher_options_set_min_score(options, 0.1);
    aicore_loma_matcher_options_free(options);
}

TEST(LomaCApiContract, DetectorOptionsLifecycle) {
    aicore_loma_detector_options* detector_options =
            aicore_loma_detector_options_new();
    EXPECT_NE(detector_options, nullptr);
    aicore_loma_detector_options_set_device(detector_options, "cpu");
    aicore_loma_detector_options_set_threads(detector_options, 1);
    aicore_loma_detector_options_set_max_keypoints(detector_options, 64);
    aicore_loma_detector_options_free(detector_options);
}

TEST(LomaCApiContract, DescriptorOptionsLifecycle) {
    aicore_loma_descriptor_options* descriptor_options =
            aicore_loma_descriptor_options_new();
    EXPECT_NE(descriptor_options, nullptr);
    aicore_loma_descriptor_options_set_device(descriptor_options, "cpu");
    aicore_loma_descriptor_options_set_threads(descriptor_options, 1);
    aicore_loma_descriptor_options_free(descriptor_options);
}

TEST(LomaCApiContract, CpuWarmup) {
    EXPECT_EQ(aicore_loma_warmup_backend("cpu"), 0);
}
