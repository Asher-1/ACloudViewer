// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "aicore/depth_capi.h"
#include "tests/common/test_macros.hpp"
#include "tests/common/validation_probe.hpp"

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
    aicore_depth_dense_result dense{};
    if (aicore_depth_depth_dense(ctx, image, &dense) != 0 || !dense.depth ||
        dense.height <= 0 || dense.width <= 0) {
        aicore_depth_dense_result_free(&dense);
        return false;
    }
    result.height = dense.height;
    result.width = dense.width;
    result.metric = dense.is_metric;
    std::copy(dense.ext, dense.ext + 12, result.ext);
    std::copy(dense.intr, dense.intr + 9, result.intr);
    const size_t count = static_cast<size_t>(result.height) * result.width;
    result.depth.assign(dense.depth, dense.depth + count);
    if (dense.conf) {
        result.confidence.assign(dense.conf, dense.conf + count);
    }
    aicore_depth_dense_result_free(&dense);
    return true;
}

// Relative MAE per compared tensor, emitted as one JSON line so the
// validation runner can gate the numbers structurally (validation_manifest
// depth-backend-parity row) instead of trusting only the exit code.
std::vector<std::pair<std::string, double>> g_metrics;

bool Compare(const char* name,
             const std::vector<float>& reference,
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
    std::fprintf(stderr, "%s relative MAE %.6f (limit %.6f)\n", name,
                 relative_mae, relative_mae_limit);
    g_metrics.emplace_back(std::string(name) + "_relative_mae", relative_mae);
    return relative_mae <= relative_mae_limit;
}

// Byte hash of every typed output the parity compares. The runner's exact
// fingerprint policy requires both sides to be byte-identical across probe
// processes (and against the baseline), on top of the MAE thresholds.
uint64_t HashDense(const DenseResult& r) {
    uint64_t h =
            aicore::test::fnv1a(r.depth.data(), r.depth.size() * sizeof(float));
    h = aicore::test::fnv1aAppend(h, r.confidence.data(),
                                  r.confidence.size() * sizeof(float));
    h = aicore::test::fnv1aAppend(h, r.ext, sizeof(r.ext));
    return aicore::test::fnv1aAppend(h, r.intr, sizeof(r.intr));
}

void PrintMetricsJson(const DenseResult& reference, const DenseResult& got) {
    std::printf(
            "{\"cpu_output_hash\":\"%016llx\","
            "\"device_output_hash\":\"%016llx\"",
            static_cast<unsigned long long>(HashDense(reference)),
            static_cast<unsigned long long>(HashDense(got)));
    for (const auto& metric : g_metrics) {
        std::printf(",\"%s\":%.9g", metric.first.c_str(), metric.second);
    }
    std::printf("}\n");
    std::fflush(stdout);
}

aicore_depth_ctx* Load(const char* anyview_or_model,
                       const char* metric,
                       const char* device) {
    aicore_depth_options* opts = aicore_depth_options_new();
    if (!opts) return nullptr;
    aicore_depth_options_set_device(opts, device);
    aicore_depth_ctx* ctx =
            metric && metric[0]
                    ? aicore_depth_load_nested_opts(anyview_or_model, metric,
                                                    opts)
                    : aicore_depth_load_opts(anyview_or_model, opts);
    aicore_depth_options_free(opts);
    return ctx;
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
    AICORE_CHECK(Compare("depth", reference.depth, got.depth, 0.015f));
    if (!reference.confidence.empty() || !got.confidence.empty()) {
        AICORE_CHECK(Compare("confidence", reference.confidence, got.confidence,
                             0.02f));
    }

    std::vector<float> ref_pose(reference.ext, reference.ext + 12);
    std::vector<float> got_pose(got.ext, got.ext + 12);
    std::vector<float> ref_intr(reference.intr, reference.intr + 9);
    std::vector<float> got_intr(got.intr, got.intr + 9);
    AICORE_CHECK(Compare("pose", ref_pose, got_pose, 0.02f));
    AICORE_CHECK(Compare("intrinsics", ref_intr, got_intr, 0.02f));
    PrintMetricsJson(reference, got);

    aicore_depth_free(cpu);
    aicore_depth_free(accelerator);
    return failures == 0 ? 0 : 1;
}
