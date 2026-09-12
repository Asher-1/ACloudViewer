// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// ViT-L token gate for the private DeDoDe-G lowering. This target links the
// AICore white-box library and is the only caller permitted to load G before
// its public C API acceptance gate is satisfied.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include "tasks/loma/descriptor.hpp"

namespace {

constexpr char kMagic[] = "LOMDGT1";
constexpr uint32_t kWidth = 784;
constexpr uint32_t kHeight = 784;
constexpr uint32_t kTokens = 3137;
constexpr uint32_t kDimension = 1024;

template <typename T>
bool Read(std::ifstream& stream, T* value) {
    return static_cast<bool>(
            stream.read(reinterpret_cast<char*>(value), sizeof(*value)));
}

struct Reference {
    std::vector<uint8_t> rgb;
    std::vector<float> tokens;
};

bool LoadReference(const char* path, Reference* reference) {
    std::ifstream stream(path, std::ios::binary);
    char magic[sizeof(kMagic)]{};
    uint32_t version = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tokens = 0;
    uint32_t dimension = 0;
    uint32_t rgb_bytes = 0;
    if (!stream.read(magic, sizeof(magic)) ||
        std::memcmp(magic, kMagic, sizeof(kMagic)) != 0 ||
        !Read(stream, &version) || !Read(stream, &width) ||
        !Read(stream, &height) || !Read(stream, &tokens) ||
        !Read(stream, &dimension) || !Read(stream, &rgb_bytes) ||
        version != 1 || width != kWidth || height != kHeight ||
        tokens != kTokens || dimension != kDimension ||
        rgb_bytes != kWidth * kHeight * 3) {
        return false;
    }
    reference->rgb.resize(rgb_bytes);
    reference->tokens.resize(static_cast<size_t>(tokens) * dimension);
    return static_cast<bool>(
                   stream.read(reinterpret_cast<char*>(reference->rgb.data()),
                               reference->rgb.size())) &&
           static_cast<bool>(stream.read(
                   reinterpret_cast<char*>(reference->tokens.data()),
                   reference->tokens.size() * sizeof(float))) &&
           stream.peek() == std::ifstream::traits_type::eof();
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_G_GGUF");
    const char* fixture =
            std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_G_TOKEN_REFERENCE");
    const char* device = std::getenv("AICORE_TEST_LOMA_TOKEN_REFERENCE_DEVICE");
    if (model == nullptr || fixture == nullptr || model[0] == '\0' ||
        fixture[0] == '\0') {
        std::fprintf(stderr,
                     "SKIP: set AICORE_TEST_LOMA_DESCRIPTOR_G_GGUF and "
                     "AICORE_TEST_LOMA_DESCRIPTOR_G_TOKEN_REFERENCE\n");
        return 77;
    }
    Reference reference;
    if (!LoadReference(fixture, &reference)) {
        std::fprintf(stderr, "invalid DeDoDe-G ONNX token-trace fixture\n");
        return 1;
    }
    aicore::loma::DescriptorOptions options;
    options.device = device != nullptr && device[0] != '\0' ? device : "cpu";
    options.allow_ungated_g_for_validation = true;
    options.trace_g_tokens_for_validation = true;
    if (const char* block =
                std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_G_TRACE_BLOCK")) {
        char* end = nullptr;
        const long value = std::strtol(block, &end, 10);
        if (end == block || *end != '\0' || value < -3 || value >= 24) {
            std::fprintf(stderr, "invalid DeDoDe-G trace block selector\n");
            return 1;
        }
        options.trace_g_block_for_validation = static_cast<int32_t>(value);
    }
    aicore::loma::Descriptor descriptor;
    if (!descriptor.Load(model, options)) {
        std::fprintf(stderr, "DeDoDe-G load failed: %s\n",
                     descriptor.error().c_str());
        return 1;
    }
    const aicore_loma_rgb_image image = {
            reference.rgb.data(), static_cast<int32_t>(kWidth),
            static_cast<int32_t>(kHeight), static_cast<int32_t>(kWidth * 3)};
    const aicore_loma_keypoint point = {0.0f, 0.0f};
    std::vector<float> output;
    if (!descriptor.Describe(image, &point, 1, static_cast<int32_t>(kWidth),
                             static_cast<int32_t>(kHeight), &output) ||
        output.size() != reference.tokens.size()) {
        std::fprintf(stderr, "DeDoDe-G token trace failed: %s\n",
                     descriptor.error().c_str());
        return 1;
    }
    float max_absolute_error = 0.0f;
    double squared_error = 0.0;
    double reference_squared = 0.0;
    for (size_t index = 0; index < output.size(); ++index) {
        const float diff = output[index] - reference.tokens[index];
        max_absolute_error = std::max(max_absolute_error, std::abs(diff));
        squared_error += static_cast<double>(diff) * diff;
        reference_squared += static_cast<double>(reference.tokens[index]) *
                             reference.tokens[index];
    }
    const double relative_l2 =
            std::sqrt(squared_error / std::max(reference_squared, 1e-30));
    std::printf(
            "{\"suite\":\"loma-dedode-g-token-reference\",\"device\":\"%s\","
            "\"tokens\":%u,"
            "\"dimension\":%u,\"max_absolute_error\":%.8g,\"relative_l2\":%.8g}"
            "\n",
            options.device.c_str(), kTokens, kDimension, max_absolute_error,
            relative_l2);
    if (max_absolute_error > 2e-3f || relative_l2 > 2e-4) {
        std::fprintf(stderr, "DeDoDe-G ViT-L token numerical gate failed\n");
        return 1;
    }
    return 0;
}
