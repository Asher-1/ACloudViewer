// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "aicore/depth_capi.h"
#include "common/test_macros.hpp"

namespace {

struct DenseResult {
    int height = 0;
    int width = 0;
    int metric = 0;
    std::vector<float> depth;
    std::vector<float> confidence;
    float ext[12] = {};
    float intr[9] = {};
};

bool RunDense(aicore_depth_ctx* ctx, const char* image, DenseResult& result) {
    float* depth = nullptr;
    float* confidence = nullptr;
    float* sky = nullptr;
    if (aicore_depth_depth_dense(ctx, image, &result.height, &result.width,
                                 &depth, &confidence, &sky, result.ext,
                                 result.intr, &result.metric) != 0 ||
        !depth || result.height <= 0 || result.width <= 0) {
        aicore_depth_free_floats(depth);
        aicore_depth_free_floats(confidence);
        aicore_depth_free_floats(sky);
        return false;
    }
    const size_t count = static_cast<size_t>(result.height) * result.width;
    result.depth.assign(depth, depth + count);
    if (confidence) result.confidence.assign(confidence, confidence + count);
    aicore_depth_free_floats(depth);
    aicore_depth_free_floats(confidence);
    aicore_depth_free_floats(sky);
    return true;
}

bool Compare(const std::vector<float>& reference,
             const std::vector<float>& got,
             float relative_mae_limit) {
    if (reference.size() != got.size() || reference.empty()) return false;
    double absolute_error = 0.0;
    double reference_magnitude = 0.0;
    for (size_t i = 0; i < reference.size(); ++i) {
        if (!std::isfinite(reference[i]) || !std::isfinite(got[i]))
            return false;
        absolute_error += std::abs(static_cast<double>(reference[i]) - got[i]);
        reference_magnitude += std::abs(static_cast<double>(reference[i]));
    }
    const double relative_mae =
            absolute_error / std::max(reference_magnitude, 1.0e-6);
    std::fprintf(stderr, "relative MAE %.6f (limit %.6f)\n", relative_mae,
                 relative_mae_limit);
    return relative_mae <= relative_mae_limit;
}

aicore_depth_ctx* Load(const char* anyview_or_model,
                       const char* metric,
                       const char* device) {
    return metric && metric[0]
                   ? aicore_depth_load_nested_device(anyview_or_model, metric,
                                                     0, device)
                   : aicore_depth_load_device(anyview_or_model, 0, device);
}

}  // namespace

static int failures = 0;

int main() {
    const char* gguf = std::getenv("AICORE_TEST_DEPTH_GGUF");
    const char* image = std::getenv("AICORE_TEST_DEPTH_IMAGE");
    const char* device = std::getenv("AICORE_TEST_DEVICE");
    const char* metric = std::getenv("AICORE_TEST_DEPTH_METRIC_GGUF");
    if (!gguf || !gguf[0] || !image || !image[0] || !device || !device[0] ||
        std::strcmp(device, "cpu") == 0) {
        return 77;
    }

    aicore_depth_ctx* cpu = Load(gguf, metric, "cpu");
    aicore_depth_ctx* accelerator = Load(gguf, metric, device);
    AICORE_CHECK(cpu != nullptr);
    AICORE_CHECK(accelerator != nullptr);
    if (!cpu || !accelerator) {
        aicore_depth_free(cpu);
        aicore_depth_free(accelerator);
        return 1;
    }

    // A fixed task resolution makes a backend result comparable. This catches
    // hidden backend-specific input caps before COLMAP can mask the mismatch.
    constexpr int kRequestedTaskTarget = 1512;
    constexpr int kParityResizeTarget = 504;
    AICORE_CHECK(aicore_depth_cap_img_resize_target(
                         cpu, kRequestedTaskTarget) == kRequestedTaskTarget);
    AICORE_CHECK(aicore_depth_cap_img_resize_target(accelerator,
                                                    kRequestedTaskTarget) ==
                 kRequestedTaskTarget);
    aicore_depth_set_img_resize_target(cpu, kParityResizeTarget);
    aicore_depth_set_img_resize_target(accelerator, kParityResizeTarget);

    DenseResult reference;
    DenseResult got;
    AICORE_CHECK(RunDense(cpu, image, reference));
    AICORE_CHECK(RunDense(accelerator, image, got));
    AICORE_CHECK(reference.height == got.height &&
                 reference.width == got.width);
    AICORE_CHECK(reference.metric == got.metric);
    AICORE_CHECK(Compare(reference.depth, got.depth, 0.015f));
    if (!reference.confidence.empty() || !got.confidence.empty()) {
        AICORE_CHECK(Compare(reference.confidence, got.confidence, 0.02f));
    }

    std::vector<float> ref_pose(reference.ext, reference.ext + 12);
    std::vector<float> got_pose(got.ext, got.ext + 12);
    std::vector<float> ref_intr(reference.intr, reference.intr + 9);
    std::vector<float> got_intr(got.intr, got.intr + 9);
    AICORE_CHECK(Compare(ref_pose, got_pose, 0.02f));
    AICORE_CHECK(Compare(ref_intr, got_intr, 0.02f));

    aicore_depth_free(cpu);
    aicore_depth_free(accelerator);
    return failures == 0 ? 0 : 1;
}
