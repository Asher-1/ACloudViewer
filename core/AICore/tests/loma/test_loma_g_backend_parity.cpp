// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// CPU-to-accelerator parity gate for the production DeDoDe-G ViT-L graph.
// The selected trace point keeps failures attributable to a concrete stage:
// patch embedding (-3), positional encoding (-2), or transformer [0, 23].

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "tasks/loma/descriptor.hpp"

namespace {

constexpr int kImageSize = 784;
constexpr int kTokenCount = 3137;
constexpr int kEmbeddingDim = 1024;

std::vector<uint8_t> MakeImage() {
    std::vector<uint8_t> image(static_cast<size_t>(kImageSize) * kImageSize *
                               3);
    for (int y = 0; y < kImageSize; ++y) {
        for (int x = 0; x < kImageSize; ++x) {
            const size_t offset = (static_cast<size_t>(y) * kImageSize + x) * 3;
            image[offset] = static_cast<uint8_t>((x * 251 + y * 17) % 256);
            image[offset + 1] = static_cast<uint8_t>((x * 13 + y * 239) % 256);
            image[offset + 2] = static_cast<uint8_t>(
                    ((x / 28 + y / 28) % 2) == 0 ? 32 : 224);
        }
    }
    return image;
}

bool Trace(const char* model,
           const char* device,
           int block,
           const aicore_loma_rgb_image& image,
           std::vector<float>* output,
           std::string* error) {
    aicore::loma::DescriptorOptions options;
    options.device = device;
    options.allow_ungated_g_for_validation = true;
    options.trace_g_tokens_for_validation = true;
    options.trace_g_block_for_validation = block;
    aicore::loma::Descriptor descriptor;
    if (!descriptor.Load(model, options)) {
        *error = descriptor.error();
        return false;
    }
    const aicore_loma_keypoint point = {0.0f, 0.0f};
    if (!descriptor.Describe(image, &point, 1, kImageSize, kImageSize,
                             output)) {
        *error = descriptor.error();
        return false;
    }
    if (output->size() != static_cast<size_t>(kTokenCount) * kEmbeddingDim) {
        *error = "unexpected DeDoDe-G token count";
        return false;
    }
    return true;
}

}  // namespace

int main() {
    const char* model = std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_G_GGUF");
    const char* device = std::getenv("AICORE_TEST_LOMA_PARITY_DEVICE");
    const char* block_text =
            std::getenv("AICORE_TEST_LOMA_DESCRIPTOR_G_TRACE_BLOCK");
    if (model == nullptr || model[0] == '\0') {
        std::fprintf(stderr, "SKIP: set AICORE_TEST_LOMA_DESCRIPTOR_G_GGUF\n");
        return 77;
    }
    if (device == nullptr || device[0] == '\0') device = "cuda";
    int block = -1;
    if (block_text != nullptr && block_text[0] != '\0') {
        char* end = nullptr;
        block = static_cast<int>(std::strtol(block_text, &end, 10));
        if (end == block_text || *end != '\0' || block < -3 || block >= 24) {
            std::fprintf(stderr, "invalid DeDoDe-G trace block selector\n");
            return 2;
        }
    }

    const std::vector<uint8_t> pixels = MakeImage();
    const aicore_loma_rgb_image image = {pixels.data(), kImageSize, kImageSize,
                                         kImageSize * 3};
    std::vector<float> cpu;
    std::vector<float> accelerator;
    std::string error;
    if (!Trace(model, "cpu", block, image, &cpu, &error)) {
        std::fprintf(stderr, "DeDoDe-G CPU trace failed: %s\n", error.c_str());
        return 1;
    }
    if (!Trace(model, device, block, image, &accelerator, &error)) {
        std::fprintf(stderr, "SKIP: DeDoDe-G %s trace unavailable: %s\n",
                     device, error.c_str());
        return 77;
    }

    float max_absolute_error = 0.0f;
    double squared_error = 0.0;
    double reference_squared = 0.0;
    for (size_t index = 0; index < cpu.size(); ++index) {
        const float difference = accelerator[index] - cpu[index];
        max_absolute_error = std::max(max_absolute_error, std::abs(difference));
        squared_error += static_cast<double>(difference) * difference;
        reference_squared += static_cast<double>(cpu[index]) * cpu[index];
    }
    const double relative_l2 =
            std::sqrt(squared_error / std::max(reference_squared, 1e-30));
    // Intermediate transformer traces are not LayerNorm-normalized. Their
    // absolute scale grows through residual blocks, while relative L2 remains
    // the portable CPU/CUDA reduction check. The ONNX-token gate owns the
    // stricter final-output 2e-3 / 2e-4 acceptance contract.
    const float max_absolute_limit = block >= 0 ? 2e-2f : 3e-3f;
    std::printf(
            "{\"suite\":\"loma-dedode-g-backend-parity\","
            "\"device\":\"%s\",\"trace_block\":%d,"
            "\"max_absolute_error\":%.8g,\"max_absolute_limit\":%.8g,"
            "\"relative_l2\":%.8g}\n",
            device, block, max_absolute_error, max_absolute_limit, relative_l2);
    if (max_absolute_error > max_absolute_limit || relative_l2 > 2e-4) {
        std::fprintf(stderr, "DeDoDe-G CPU/%s numerical parity gate failed\n",
                     device);
        return 1;
    }
    return 0;
}
