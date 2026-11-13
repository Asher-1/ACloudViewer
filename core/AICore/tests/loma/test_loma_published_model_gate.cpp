// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Per-artifact gate for the published GGUF-only LoMa catalog. The selected
// artifact is supplied from the shared cache by validation_manifest.json.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aicore/loma_capi.h"

namespace {

std::vector<uint8_t> MakePattern(const int width, const int height) {
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
            rgb[offset] = static_cast<uint8_t>((x * 251 + y * 17) % 256);
            rgb[offset + 1] = static_cast<uint8_t>((x * 13 + y * 239) % 256);
            rgb[offset + 2] =
                    static_cast<uint8_t>(((x / 24 + y / 24) % 2) ? 224 : 32);
        }
    }
    return rgb;
}

bool IsFinite(const float* values, const size_t count) {
    for (size_t index = 0; index < count; ++index) {
        if (!std::isfinite(values[index])) return false;
    }
    return true;
}

uint64_t HashBytes(const void* data, const size_t size, uint64_t hash) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

int RunDetector(const char* model, const char* device) {
    constexpr int kWidth = 784;
    constexpr int kHeight = 784;
    const std::vector<uint8_t> rgb = MakePattern(kWidth, kHeight);
    const aicore_loma_rgb_image image = {rgb.data(), kWidth, kHeight,
                                         kWidth * 3};
    aicore_loma_detector_options* options = aicore_loma_detector_options_new();
    aicore_loma_detector_options_set_device(options, device);
    aicore_loma_detector_options_set_max_keypoints(options, 128);
    aicore_loma_detector_ctx* context =
            aicore_loma_detector_load(model, options);
    aicore_loma_detector_options_free(options);
    aicore_loma_detected_features features{};
    const int rc =
            aicore_loma_detector_is_ready(context)
                    ? aicore_loma_detector_run(context, &image, &features)
                    : -1;
    const bool valid =
            rc == 0 && features.count > 0 &&
            IsFinite(features.scores, static_cast<size_t>(features.count));
    uint64_t output_hash = 1469598103934665603ULL;
    if (valid) {
        output_hash = HashBytes(features.keypoints,
                                static_cast<size_t>(features.count) *
                                        sizeof(*features.keypoints),
                                output_hash);
        output_hash = HashBytes(
                features.scores,
                static_cast<size_t>(features.count) * sizeof(*features.scores),
                output_hash);
        std::printf(
                "{\"suite\":\"loma-published-model\","
                "\"role\":\"detector\",\"count\":%d,"
                "\"output_hash\":\"%016llx\"}\n",
                features.count, static_cast<unsigned long long>(output_hash));
    } else {
        std::fprintf(stderr, "LoMa detector artifact failed: %s\n",
                     aicore_loma_detector_last_error(context));
    }
    aicore_loma_detected_features_free(&features);
    aicore_loma_detector_free(context);
    return valid ? 0 : 1;
}

int RunDescriptor(const char* model, const char* device) {
    constexpr int kWidth = 784;
    constexpr int kHeight = 784;
    const std::vector<uint8_t> rgb = MakePattern(kWidth, kHeight);
    const aicore_loma_rgb_image image = {rgb.data(), kWidth, kHeight,
                                         kWidth * 3};
    const aicore_loma_keypoint keypoints[] = {{96.0f, 96.0f},
                                              {256.0f, 160.0f},
                                              {480.0f, 420.0f},
                                              {640.0f, 640.0f}};
    aicore_loma_descriptor_options* options =
            aicore_loma_descriptor_options_new();
    aicore_loma_descriptor_options_set_device(options, device);
    aicore_loma_descriptor_ctx* context =
            aicore_loma_descriptor_load(model, options);
    aicore_loma_descriptor_options_free(options);
    aicore_loma_described_features descriptors{};
    const int rc =
            aicore_loma_descriptor_is_ready(context)
                    ? aicore_loma_descriptor_run(context, &image, keypoints, 4,
                                                 kWidth, kHeight, &descriptors)
                    : -1;
    const bool valid = rc == 0 && descriptors.count == 4 &&
                       (descriptors.descriptor_dim == 128 ||
                        descriptors.descriptor_dim == 256) &&
                       IsFinite(descriptors.descriptors,
                                static_cast<size_t>(descriptors.count) *
                                        descriptors.descriptor_dim);
    uint64_t output_hash = 1469598103934665603ULL;
    if (valid) {
        output_hash =
                HashBytes(descriptors.descriptors,
                          static_cast<size_t>(descriptors.count) *
                                  descriptors.descriptor_dim * sizeof(float),
                          output_hash);
        std::printf(
                "{\"suite\":\"loma-published-model\","
                "\"role\":\"descriptor\",\"dimension\":%d,"
                "\"output_hash\":\"%016llx\"}\n",
                descriptors.descriptor_dim,
                static_cast<unsigned long long>(output_hash));
    } else {
        std::fprintf(stderr, "LoMa descriptor artifact failed: %s\n",
                     aicore_loma_descriptor_last_error(context));
    }
    aicore_loma_described_features_free(&descriptors);
    aicore_loma_descriptor_free(context);
    return valid ? 0 : 1;
}

int RunMatcher(const char* model, const char* device, const int dimension) {
    constexpr int kCount = 32;
    std::vector<aicore_loma_keypoint> keypoints(kCount);
    std::vector<float> descriptors(static_cast<size_t>(kCount) * dimension);
    for (int index = 0; index < kCount; ++index) {
        keypoints[index] = {32.0f + static_cast<float>(index * 13),
                            48.0f + static_cast<float>(index * 9)};
        float* descriptor =
                descriptors.data() + static_cast<size_t>(index) * dimension;
        double squared_norm = 0.0;
        for (int dim = 0; dim < dimension; ++dim) {
            descriptor[dim] = std::sin(0.013f * (index + 1) * (dim + 3));
            squared_norm += descriptor[dim] * descriptor[dim];
        }
        const float inverse_norm =
                1.0f / static_cast<float>(std::sqrt(squared_norm));
        for (int dim = 0; dim < dimension; ++dim) {
            descriptor[dim] *= inverse_norm;
        }
    }
    const aicore_loma_features features = {
            keypoints.data(), kCount, descriptors.data(), dimension, 640, 480};
    aicore_loma_matcher_options* options = aicore_loma_matcher_options_new();
    aicore_loma_matcher_options_set_device(options, device);
    aicore_loma_matcher_options_set_min_score(options, 0.0);
    aicore_loma_matcher_ctx* context = aicore_loma_matcher_load(model, options);
    aicore_loma_matcher_options_free(options);
    aicore_loma_match* matches = nullptr;
    int32_t match_count = 0;
    const int rc =
            aicore_loma_matcher_is_ready(context)
                    ? aicore_loma_matcher_run(context, &features, &features,
                                              &matches, &match_count)
                    : -1;
    bool valid = rc == 0 && match_count > 0;
    for (int32_t index = 0; valid && index < match_count; ++index) {
        valid = matches[index].idx0 >= 0 && matches[index].idx0 < kCount &&
                matches[index].idx1 >= 0 && matches[index].idx1 < kCount &&
                std::isfinite(matches[index].score);
    }
    uint64_t output_hash = 1469598103934665603ULL;
    if (valid) {
        output_hash = HashBytes(
                matches, static_cast<size_t>(match_count) * sizeof(*matches),
                output_hash);
        std::printf(
                "{\"suite\":\"loma-published-model\","
                "\"role\":\"matcher\",\"dimension\":%d,"
                "\"matches\":%d,\"output_hash\":\"%016llx\"}\n",
                dimension, match_count,
                static_cast<unsigned long long>(output_hash));
    } else {
        std::fprintf(stderr, "LoMa matcher artifact failed: %s\n",
                     aicore_loma_matcher_last_error(context));
    }
    aicore_loma_free_matches(matches);
    aicore_loma_matcher_free(context);
    return valid ? 0 : 1;
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_LOMA_GGUF");
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    if (model == nullptr || model[0] == '\0') {
        std::fprintf(stderr, "SKIP: set AICORE_TEST_LOMA_GGUF\n");
        return 77;
    }
    if (device == nullptr || device[0] == '\0') device = "cpu";
    const std::string path(model);
    if (path.find("loma_detector.") != std::string::npos) {
        return RunDetector(model, device);
    }
    if (path.find("loma_descriptor_") != std::string::npos) {
        return RunDescriptor(model, device);
    }
    if (path.find("loma_matcher_") != std::string::npos) {
        const int dimension =
                path.find("loma_matcher_B128.") != std::string::npos ? 128
                                                                     : 256;
        return RunMatcher(model, device, dimension);
    }
    std::fprintf(stderr, "unrecognized published LoMa model: %s\n", model);
    return 2;
}
