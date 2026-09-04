// Default COLMAP LoMa model-pack gate: DaD + DeDoDe-G + LoMa-B, GGUF only.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "aicore/loma_capi.h"

namespace {

std::vector<uint8_t> MakePattern(const int width, const int height) {
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int checker = ((x / 28) + (y / 28)) % 2;
            const size_t offset = (static_cast<size_t>(y) * width + x) * 3;
            rgb[offset + 0] = static_cast<uint8_t>((x * 255) / width);
            rgb[offset + 1] = static_cast<uint8_t>((y * 255) / height);
            rgb[offset + 2] = static_cast<uint8_t>(checker ? 220 : 35);
        }
    }
    return rgb;
}

uint64_t HashBytes(const void* data, const size_t size, uint64_t hash) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t index = 0; index < size; ++index) {
        hash ^= bytes[index];
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

int main() {
    const char* detector_model = std::getenv("AICORE_TEST_LOMA_DETECTOR_GGUF");
    const char* descriptor_model = std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_GGUF");
    const char* matcher_model = std::getenv("AICORE_TEST_LOMA_MATCHER_GGUF");
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    const char* expected_dimension_text =
            std::getenv("AICORE_TEST_LOMA_EXPECTED_DESCRIPTOR_DIM");
    if (detector_model == nullptr || descriptor_model == nullptr || matcher_model == nullptr ||
        detector_model[0] == '\0' || descriptor_model[0] == '\0' ||
        matcher_model[0] == '\0') {
        std::fprintf(stderr, "SKIP: set AICORE_TEST_LOMA_{DETECTOR,DESCRIPTOR,MATCHER}_GGUF\n");
        return 77;
    }
    if (device == nullptr || device[0] == '\0') device = "cpu";
    int expected_dimension = 256;
    if (expected_dimension_text != nullptr && expected_dimension_text[0] != '\0') {
        expected_dimension = std::atoi(expected_dimension_text);
    }
    if (expected_dimension != 128 && expected_dimension != 256) {
        std::fprintf(stderr, "invalid expected LoMa descriptor dimension: %d\n",
                     expected_dimension);
        return 2;
    }

    constexpr int kWidth = 784;
    constexpr int kHeight = 784;
    const std::vector<uint8_t> rgb = MakePattern(kWidth, kHeight);
    const aicore_loma_rgb_image image = {rgb.data(), kWidth, kHeight, kWidth * 3};

    aicore_loma_detector_options* detector_options = aicore_loma_detector_options_new();
    aicore_loma_detector_options_set_device(detector_options, device);
    aicore_loma_detector_options_set_max_keypoints(detector_options, 256);
    aicore_loma_detector_ctx* detector =
            aicore_loma_detector_load(detector_model, detector_options);
    aicore_loma_detector_options_free(detector_options);
    aicore_loma_descriptor_options* descriptor_options =
            aicore_loma_descriptor_options_new();
    aicore_loma_descriptor_options_set_device(descriptor_options, device);
    aicore_loma_descriptor_ctx* descriptor =
            aicore_loma_descriptor_load(descriptor_model, descriptor_options);
    aicore_loma_descriptor_options_free(descriptor_options);
    aicore_loma_matcher_options* matcher_options = aicore_loma_matcher_options_new();
    aicore_loma_matcher_options_set_device(matcher_options, device);
    aicore_loma_matcher_options_set_min_score(matcher_options, 0.1);
    aicore_loma_matcher_ctx* matcher =
            aicore_loma_matcher_load(matcher_model, matcher_options);
    aicore_loma_matcher_options_free(matcher_options);
    if (!aicore_loma_detector_is_ready(detector) ||
        !aicore_loma_descriptor_is_ready(descriptor) ||
        !aicore_loma_matcher_is_ready(matcher)) {
        std::fprintf(stderr, "LoMa model-pack load failed: detector=%s descriptor=%s matcher=%s\n",
                     aicore_loma_detector_last_error(detector),
                     aicore_loma_descriptor_last_error(descriptor),
                     aicore_loma_matcher_last_error(matcher));
        aicore_loma_matcher_free(matcher);
        aicore_loma_descriptor_free(descriptor);
        aicore_loma_detector_free(detector);
        return 1;
    }

    aicore_loma_detected_features points{};
    aicore_loma_described_features descriptors{};
    aicore_loma_match* matches = nullptr;
    int32_t match_count = 0;
    const int detect_rc = aicore_loma_detector_run(detector, &image, &points);
    const int describe_rc = detect_rc == 0
            ? aicore_loma_descriptor_run(descriptor, &image, points.keypoints, points.count,
                                         kWidth, kHeight, &descriptors)
            : -1;
    const aicore_loma_features features = {points.keypoints, points.count,
                                           descriptors.descriptors,
                                           descriptors.descriptor_dim,
                                           points.image_width, points.image_height};
    const int match_rc = describe_rc == 0
            ? aicore_loma_matcher_run(matcher, &features, &features, &matches, &match_count)
            : -1;
    const bool valid = detect_rc == 0 && points.count > 0 && describe_rc == 0 &&
                       descriptors.count == points.count &&
                       descriptors.descriptor_dim == expected_dimension &&
                       match_rc == 0 && match_count > 0;
    uint64_t output_hash = 1469598103934665603ULL;
    if (valid) {
        output_hash = HashBytes(points.keypoints,
                                static_cast<size_t>(points.count) * sizeof(*points.keypoints),
                                output_hash);
        output_hash = HashBytes(descriptors.descriptors,
                                static_cast<size_t>(descriptors.count) *
                                        descriptors.descriptor_dim * sizeof(float),
                                output_hash);
        output_hash = HashBytes(matches,
                                static_cast<size_t>(match_count) * sizeof(*matches),
                                output_hash);
    }
    std::printf("{\"suite\":\"loma-default-triplet\",\"keypoints\":%d,"
                "\"descriptor_dim\":%d,\"matches\":%d,\"output_hash\":\"%016llx\"}\n",
                points.count, descriptors.descriptor_dim, match_count,
                static_cast<unsigned long long>(output_hash));
    aicore_loma_free_matches(matches);
    aicore_loma_described_features_free(&descriptors);
    aicore_loma_detected_features_free(&points);
    aicore_loma_matcher_free(matcher);
    aicore_loma_descriptor_free(descriptor);
    aicore_loma_detector_free(detector);
    if (!valid) {
        std::fprintf(stderr, "LoMa default model-pack functional gate failed\n");
        return 1;
    }
    return 0;
}
