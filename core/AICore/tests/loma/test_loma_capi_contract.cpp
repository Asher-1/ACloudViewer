// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>

#include "aicore/aicore.h"
#include "tests/common/test_macros.hpp"

static int failures = 0;

int main() {
    AICORE_CHECK(aicore_loma_abi_version() == 4);
    AICORE_CHECK(aicore_loma_model_count() == 22);
    for (int role = AICORE_LOMA_MODEL_ROLE_DETECTOR;
         role <= AICORE_LOMA_MODEL_ROLE_MATCHER; ++role) {
        const aicore_loma_model_entry* model = aicore_loma_model_by_role(
                static_cast<aicore_loma_model_role>(role));
        AICORE_CHECK(model != nullptr);
        AICORE_CHECK(model->filename != nullptr && model->filename[0] != '\0');
        AICORE_CHECK(model->download_url != nullptr &&
                     model->download_url[0] != '\0');
        AICORE_CHECK(model->role == role);
    }
    for (int variant = AICORE_LOMA_MODEL_VARIANT_DAD;
         variant <= AICORE_LOMA_MODEL_VARIANT_MATCHER_G; ++variant) {
        const aicore_loma_model_entry* model = aicore_loma_model_by_variant(
                static_cast<aicore_loma_model_variant>(variant));
        AICORE_CHECK(model != nullptr);
        AICORE_CHECK(model->variant == variant);
    }
    char* model_cache = aicore_loma_model_cache_dir();
    AICORE_CHECK(model_cache != nullptr);
    std::free(model_cache);
    aicore_loma_detector_free(nullptr);
    aicore_loma_detector_options_free(nullptr);
    aicore_loma_detected_features empty{};
    aicore_loma_detected_features_free(&empty);
    AICORE_CHECK(aicore_loma_detector_load(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_loma_detector_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_loma_detector_last_error(nullptr) != nullptr);
    AICORE_CHECK(aicore_loma_detector_run(nullptr, nullptr, nullptr) != 0);
    aicore_loma_descriptor_free(nullptr);
    aicore_loma_descriptor_options_free(nullptr);
    aicore_loma_described_features empty_descriptors{};
    aicore_loma_described_features_free(&empty_descriptors);
    AICORE_CHECK(aicore_loma_descriptor_load(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_loma_descriptor_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_loma_descriptor_last_error(nullptr) != nullptr);
    AICORE_CHECK(aicore_loma_descriptor_run(nullptr, nullptr, nullptr, 0, 0, 0,
                                            nullptr) != 0);
    aicore_loma_matcher_free(nullptr);
    aicore_loma_free_matches(nullptr);
    aicore_loma_matcher_options_free(nullptr);
    AICORE_CHECK(aicore_loma_matcher_load(nullptr, nullptr) == nullptr);
    AICORE_CHECK(aicore_loma_matcher_is_ready(nullptr) == 0);
    AICORE_CHECK(aicore_loma_matcher_last_error(nullptr) != nullptr);
    AICORE_CHECK(aicore_loma_matcher_run(nullptr, nullptr, nullptr, nullptr,
                                         nullptr) != 0);

    aicore_loma_matcher_options* options = aicore_loma_matcher_options_new();
    AICORE_CHECK(options != nullptr);
    aicore_loma_matcher_options_set_device(options, "cpu");
    aicore_loma_matcher_options_set_threads(options, 1);
    aicore_loma_matcher_options_set_min_score(options, 0.1);
    aicore_loma_matcher_options_free(options);
    aicore_loma_detector_options* detector_options =
            aicore_loma_detector_options_new();
    AICORE_CHECK(detector_options != nullptr);
    aicore_loma_detector_options_set_device(detector_options, "cpu");
    aicore_loma_detector_options_set_threads(detector_options, 1);
    aicore_loma_detector_options_set_max_keypoints(detector_options, 64);
    aicore_loma_detector_options_free(detector_options);
    aicore_loma_descriptor_options* descriptor_options =
            aicore_loma_descriptor_options_new();
    AICORE_CHECK(descriptor_options != nullptr);
    aicore_loma_descriptor_options_set_device(descriptor_options, "cpu");
    aicore_loma_descriptor_options_set_threads(descriptor_options, 1);
    aicore_loma_descriptor_options_free(descriptor_options);
    AICORE_CHECK(aicore_loma_warmup_backend("cpu") == 0);
    return failures;
}
