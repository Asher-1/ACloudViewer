// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Full real-image LoMa-B128 gate: DaD -> DeDoDe-B -> LoMa-B128. The fixture
// is produced from pinned upstream ONNX graphs; this test uses only GGUF and
// public AICore APIs at runtime.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "aicore/loma_capi.h"

namespace {

constexpr char kMagic[] = "LOMB128";

template <typename T>
bool Read(std::ifstream& stream, T* value) {
    return static_cast<bool>(
            stream.read(reinterpret_cast<char*>(value), sizeof(*value)));
}

struct Reference {
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t count = 0;
    uint32_t descriptor_dim = 0;
    std::vector<uint8_t> rgb0;
    std::vector<uint8_t> rgb1;
    std::map<std::pair<int32_t, int32_t>, float> matches;
};

bool LoadReference(const char* path, Reference* reference) {
    std::ifstream stream(path, std::ios::binary);
    char magic[sizeof(kMagic)]{};
    uint32_t version = 0;
    uint32_t rgb_bytes = 0;
    uint32_t match_count = 0;
    if (!stream.read(magic, sizeof(magic)) ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        !Read(stream, &version) || !Read(stream, &reference->width) ||
        !Read(stream, &reference->height) || !Read(stream, &reference->count) ||
        !Read(stream, &reference->descriptor_dim) ||
        !Read(stream, &rgb_bytes) || !Read(stream, &match_count) ||
        version != 1 || reference->width != 784 || reference->height != 784 ||
        reference->count == 0 || reference->descriptor_dim != 128 ||
        match_count == 0 ||
        rgb_bytes != reference->width * reference->height * 3) {
        return false;
    }
    reference->rgb0.resize(rgb_bytes);
    reference->rgb1.resize(rgb_bytes);
    if (!stream.read(reinterpret_cast<char*>(reference->rgb0.data()),
                     reference->rgb0.size()) ||
        !stream.read(reinterpret_cast<char*>(reference->rgb1.data()),
                     reference->rgb1.size())) {
        return false;
    }
    for (uint32_t index = 0; index < match_count; ++index) {
        int32_t index0 = -1;
        int32_t index1 = -1;
        float score = 0.0f;
        if (!Read(stream, &index0) || !Read(stream, &index1) ||
            !Read(stream, &score) || index0 < 0 || index1 < 0 ||
            index0 >= static_cast<int32_t>(reference->count) ||
            index1 >= static_cast<int32_t>(reference->count) ||
            !std::isfinite(score)) {
            return false;
        }
        reference->matches[{index0, index1}] = score;
    }
    return stream.peek() == std::ifstream::traits_type::eof();
}

bool Extract(aicore_loma_detector_ctx* detector,
             aicore_loma_descriptor_ctx* descriptor,
             const std::vector<uint8_t>& rgb,
             const Reference& reference,
             aicore_loma_detected_features* points,
             aicore_loma_described_features* descriptions) {
    const aicore_loma_rgb_image image = {
            rgb.data(), static_cast<int32_t>(reference.width),
            static_cast<int32_t>(reference.height),
            static_cast<int32_t>(reference.width * 3)};
    *points = {};
    *descriptions = {};
    if (aicore_loma_detector_run(detector, &image, points) != 0 ||
        points->count != static_cast<int32_t>(reference.count)) {
        return false;
    }
    return aicore_loma_descriptor_run(descriptor, &image, points->keypoints,
                                      points->count, image.width, image.height,
                                      descriptions) == 0 &&
           descriptions->count == points->count &&
           descriptions->descriptor_dim ==
                   static_cast<int32_t>(reference.descriptor_dim);
}

}  // namespace

int main() {
    const char* detector_model = std::getenv("AICORE_TEST_LOMA_DETECTOR_GGUF");
    const char* descriptor_model =
            std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_GGUF");
    const char* matcher_model = std::getenv("AICORE_TEST_LOMA_B128_GGUF");
    const char* fixture = std::getenv("AICORE_TEST_LOMA_B128_REFERENCE");
    if (detector_model == nullptr || descriptor_model == nullptr ||
        matcher_model == nullptr || fixture == nullptr ||
        detector_model[0] == '\0' || descriptor_model[0] == '\0' ||
        matcher_model[0] == '\0' || fixture[0] == '\0') {
        std::fprintf(
                stderr,
                "SKIP: set AICORE_TEST_LOMA_{DETECTOR,DESCRIPTOR,B128}_GGUF "
                "and AICORE_TEST_LOMA_B128_REFERENCE\n");
        return 77;
    }
    Reference reference;
    if (!LoadReference(fixture, &reference)) {
        std::fprintf(stderr,
                     "invalid real-image LoMa-B128 reference fixture\n");
        return 1;
    }
    aicore_loma_detector_options* detector_options =
            aicore_loma_detector_options_new();
    aicore_loma_detector_options_set_device(detector_options, "cpu");
    aicore_loma_detector_options_set_max_keypoints(
            detector_options, static_cast<int32_t>(reference.count));
    aicore_loma_detector_ctx* detector =
            aicore_loma_detector_load(detector_model, detector_options);
    aicore_loma_detector_options_free(detector_options);
    aicore_loma_descriptor_options* descriptor_options =
            aicore_loma_descriptor_options_new();
    aicore_loma_descriptor_options_set_device(descriptor_options, "cpu");
    aicore_loma_descriptor_ctx* descriptor =
            aicore_loma_descriptor_load(descriptor_model, descriptor_options);
    aicore_loma_descriptor_options_free(descriptor_options);
    aicore_loma_matcher_options* matcher_options =
            aicore_loma_matcher_options_new();
    aicore_loma_matcher_options_set_device(matcher_options, "cpu");
    aicore_loma_matcher_options_set_min_score(matcher_options, 0.1);
    aicore_loma_matcher_ctx* matcher =
            aicore_loma_matcher_load(matcher_model, matcher_options);
    aicore_loma_matcher_options_free(matcher_options);
    if (!aicore_loma_detector_is_ready(detector) ||
        !aicore_loma_descriptor_is_ready(descriptor) ||
        !aicore_loma_matcher_is_ready(matcher)) {
        std::fprintf(stderr,
                     "LoMa-B128 load failure: detector=%s descriptor=%s "
                     "matcher=%s\n",
                     aicore_loma_detector_last_error(detector),
                     aicore_loma_descriptor_last_error(descriptor),
                     aicore_loma_matcher_last_error(matcher));
        aicore_loma_matcher_free(matcher);
        aicore_loma_descriptor_free(descriptor);
        aicore_loma_detector_free(detector);
        return 1;
    }
    aicore_loma_detected_features points0{};
    aicore_loma_detected_features points1{};
    aicore_loma_described_features descriptions0{};
    aicore_loma_described_features descriptions1{};
    const bool extracted = Extract(detector, descriptor, reference.rgb0,
                                   reference, &points0, &descriptions0) &&
                           Extract(detector, descriptor, reference.rgb1,
                                   reference, &points1, &descriptions1);
    aicore_loma_match* raw_matches = nullptr;
    int32_t match_count = 0;
    int rc = -1;
    if (extracted) {
        const aicore_loma_features image0 = {
                points0.keypoints,         points0.count,
                descriptions0.descriptors, descriptions0.descriptor_dim,
                points0.image_width,       points0.image_height};
        const aicore_loma_features image1 = {
                points1.keypoints,         points1.count,
                descriptions1.descriptors, descriptions1.descriptor_dim,
                points1.image_width,       points1.image_height};
        rc = aicore_loma_matcher_run(matcher, &image0, &image1, &raw_matches,
                                     &match_count);
    }
    std::set<std::pair<int32_t, int32_t>> predicted;
    std::map<std::pair<int32_t, int32_t>, float> predicted_scores;
    for (int32_t index = 0; index < match_count; ++index) {
        const auto pair = std::make_pair(raw_matches[index].idx0,
                                         raw_matches[index].idx1);
        if (pair.first >= 0 && pair.second >= 0 &&
            std::isfinite(raw_matches[index].score)) {
            predicted.insert(pair);
            predicted_scores[pair] = raw_matches[index].score;
        }
    }
    size_t true_positive = 0;
    float max_score_error = 0.0f;
    for (const auto& pair : predicted) {
        const auto expected = reference.matches.find(pair);
        if (expected != reference.matches.end()) {
            ++true_positive;
            max_score_error = std::max(
                    max_score_error,
                    std::abs(predicted_scores[pair] - expected->second));
        }
    }
    const double precision =
            predicted.empty()
                    ? 0.0
                    : static_cast<double>(true_positive) / predicted.size();
    const double recall =
            static_cast<double>(true_positive) / reference.matches.size();
    std::printf(
            "{\"suite\":\"loma-b128-real-e2e\",\"expected\":%zu,"
            "\"predicted\":%zu,\"tp\":%zu,\"precision\":%.8f,"
            "\"recall\":%.8f,\"max_score_error\":%.8g}\n",
            reference.matches.size(), predicted.size(), true_positive,
            precision, recall, max_score_error);
    aicore_loma_free_matches(raw_matches);
    aicore_loma_described_features_free(&descriptions0);
    aicore_loma_described_features_free(&descriptions1);
    aicore_loma_detected_features_free(&points0);
    aicore_loma_detected_features_free(&points1);
    aicore_loma_matcher_free(matcher);
    aicore_loma_descriptor_free(descriptor);
    aicore_loma_detector_free(detector);
    if (!extracted || rc != 0 || precision < 0.995 || recall < 0.995 ||
        max_score_error > 1e-3f) {
        std::fprintf(stderr,
                     "LoMa-B128 real-image end-to-end P/R gate failed\n");
        return 1;
    }
    return 0;
}
