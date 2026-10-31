// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// DeDoDe real-image gate. The fixture contains DaD-selected points and
// descriptions from the pinned upstream ONNX graphs; this executable itself
// only loads the production GGUF graph through libAICore.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "aicore/loma_capi.h"

namespace {

constexpr char kMagic[] = "LOMDDB1";

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
    std::vector<uint8_t> rgb;
    std::vector<aicore_loma_keypoint> keypoints;
    std::vector<float> descriptors;
};

bool LoadReference(const char* path, Reference* reference) {
    std::ifstream stream(path, std::ios::binary);
    char magic[sizeof(kMagic)]{};
    uint32_t version = 0;
    uint32_t rgb_bytes = 0;
    if (!stream.read(magic, sizeof(magic)) ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        !Read(stream, &version) || !Read(stream, &reference->width) ||
        !Read(stream, &reference->height) || !Read(stream, &reference->count) ||
        !Read(stream, &reference->descriptor_dim) ||
        !Read(stream, &rgb_bytes) || version != 1 || reference->width != 784 ||
        reference->height != 784 || reference->count == 0 ||
        (reference->descriptor_dim != 128 &&
         reference->descriptor_dim != 256) ||
        rgb_bytes != reference->width * reference->height * 3) {
        return false;
    }
    reference->rgb.resize(rgb_bytes);
    reference->keypoints.resize(reference->count);
    reference->descriptors.resize(static_cast<size_t>(reference->count) *
                                  reference->descriptor_dim);
    return static_cast<bool>(
                   stream.read(reinterpret_cast<char*>(reference->rgb.data()),
                               reference->rgb.size())) &&
           static_cast<bool>(stream.read(
                   reinterpret_cast<char*>(reference->keypoints.data()),
                   reference->keypoints.size() *
                           sizeof(aicore_loma_keypoint))) &&
           static_cast<bool>(stream.read(
                   reinterpret_cast<char*>(reference->descriptors.data()),
                   reference->descriptors.size() * sizeof(float))) &&
           stream.peek() == std::ifstream::traits_type::eof();
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_GGUF");
    const char* fixture = std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_REFERENCE");
    if (model == nullptr || fixture == nullptr || model[0] == '\0' ||
        fixture[0] == '\0') {
        std::fprintf(stderr,
                     "SKIP: set AICORE_TEST_LOMA_DESCRIPTOR_GGUF and "
                     "AICORE_TEST_LOMA_DESCRIPTOR_REFERENCE\n");
        return 77;
    }
    Reference reference;
    if (!LoadReference(fixture, &reference)) {
        std::fprintf(stderr, "invalid DeDoDe ONNX-reference fixture\n");
        return 1;
    }
    aicore_loma_descriptor_options* options =
            aicore_loma_descriptor_options_new();
    aicore_loma_descriptor_options_set_device(options, "cpu");
    aicore_loma_descriptor_ctx* descriptor =
            aicore_loma_descriptor_load(model, options);
    aicore_loma_descriptor_options_free(options);
    if (!aicore_loma_descriptor_is_ready(descriptor)) {
        std::fprintf(stderr, "DeDoDe load failed: %s\n",
                     aicore_loma_descriptor_last_error(descriptor));
        aicore_loma_descriptor_free(descriptor);
        return 1;
    }
    const aicore_loma_rgb_image image = {
            reference.rgb.data(), static_cast<int32_t>(reference.width),
            static_cast<int32_t>(reference.height),
            static_cast<int32_t>(reference.width * 3)};
    aicore_loma_described_features output{};
    const int rc = aicore_loma_descriptor_run(
            descriptor, &image, reference.keypoints.data(),
            static_cast<int32_t>(reference.count),
            static_cast<int32_t>(reference.width),
            static_cast<int32_t>(reference.height), &output);
    if (rc != 0 || output.count != static_cast<int32_t>(reference.count) ||
        output.descriptor_dim !=
                static_cast<int32_t>(reference.descriptor_dim)) {
        std::fprintf(stderr, "DeDoDe run failed: count=%d dim=%d error=%s\n",
                     output.count, output.descriptor_dim,
                     aicore_loma_descriptor_last_error(descriptor));
        aicore_loma_described_features_free(&output);
        aicore_loma_descriptor_free(descriptor);
        return 1;
    }
    float max_absolute_error = 0.0f;
    double squared_error = 0.0;
    double reference_squared = 0.0;
    for (size_t index = 0; index < reference.descriptors.size(); ++index) {
        const float diff =
                output.descriptors[index] - reference.descriptors[index];
        max_absolute_error = std::max(max_absolute_error, std::abs(diff));
        squared_error += static_cast<double>(diff) * diff;
        reference_squared += static_cast<double>(reference.descriptors[index]) *
                             reference.descriptors[index];
    }
    const double relative_l2 =
            std::sqrt(squared_error / std::max(reference_squared, 1e-30));
    std::printf(
            "{\"suite\":\"loma-dedode-reference\",\"count\":%u,"
            "\"dim\":%u,\"max_absolute_error\":%.8g,"
            "\"relative_l2\":%.8g}\n",
            reference.count, reference.descriptor_dim, max_absolute_error,
            relative_l2);
    aicore_loma_described_features_free(&output);
    aicore_loma_descriptor_free(descriptor);
    if (max_absolute_error > 2e-3f || relative_l2 > 2e-4) {
        std::fprintf(stderr, "DeDoDe real-image numerical gate failed\n");
        return 1;
    }
    return 0;
}
