// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/rmbg_capi.h"

namespace {

constexpr int kInputSize = 1024;
constexpr int kWarmups = 2;
constexpr int kRuns = 7;

const char* env_or(const char* primary,
                   const char* fallback,
                   const char* default_value) {
    const char* value = std::getenv(primary);
    if (value && value[0]) return value;
    value = std::getenv(fallback);
    return value && value[0] ? value : default_value;
}

uint64_t fnv1a(const uint8_t* data, size_t size) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    const char* model = std::getenv("AICORE_TEST_RMBG_GGUF");
    if (!model || !model[0]) {
        std::printf("[rmbg-perf] skipped: AICORE_TEST_RMBG_GGUF is unset\n");
        return 77;
    }
    const char* device = argc > 1 ? argv[1]
                                  : env_or("AICORE_TEST_RMBG_DEVICE",
                                           "AICORE_TEST_DEVICE", "auto");
    if (std::strcmp(device, "cpu") == 0) {
        std::printf(
                "[rmbg-perf] skipped: GPU performance test requested CPU\n");
        return 77;
    }

    aicore_rmbg_options* options = aicore_rmbg_options_new();
    if (!options) return 1;
    aicore_rmbg_options_set_device(options, device);
    aicore_rmbg_ctx* ctx = aicore_rmbg_load_opts(model, options);
    aicore_rmbg_options_free(options);
    if (!ctx || !aicore_rmbg_is_ready(ctx)) {
        std::printf("[rmbg-perf] skipped: %s\n",
                    ctx && aicore_rmbg_last_error(ctx)
                            ? aicore_rmbg_last_error(ctx)
                            : "backend or model unavailable");
        aicore_rmbg_free(ctx);
        return 77;
    }

    char* info = aicore_rmbg_info_json(ctx);
    if (!info || !std::strstr(info, "\"math_profile\":")) {
        std::fprintf(stderr, "[rmbg-perf] missing math_profile in info JSON\n");
        aicore_rmbg_free_string(info);
        aicore_rmbg_free(ctx);
        return 1;
    }
    std::printf("[rmbg-perf] info=%s\n", info);
    aicore_rmbg_free_string(info);

    std::vector<uint8_t> rgb((size_t)kInputSize * kInputSize * 3);
    for (int y = 0; y < kInputSize; ++y) {
        for (int x = 0; x < kInputSize; ++x) {
            const size_t p = ((size_t)y * kInputSize + x) * 3;
            rgb[p + 0] = static_cast<uint8_t>((x * 13 + y * 3) & 255);
            rgb[p + 1] = static_cast<uint8_t>((x * 5 + y * 11) & 255);
            rgb[p + 2] = static_cast<uint8_t>((x ^ y) & 255);
        }
    }

    std::vector<double> samples;
    samples.reserve(kRuns);
    uint64_t reference_hash = 0;
    for (int run = -kWarmups; run < kRuns; ++run) {
        uint8_t* alpha = nullptr;
        int32_t width = 0;
        int32_t height = 0;
        if (aicore_rmbg_alpha_mat_rgb(ctx, rgb.data(), kInputSize, kInputSize,
                                      &alpha, &width, &height) != 0 ||
            !alpha || width != kInputSize || height != kInputSize) {
            std::fprintf(stderr, "[rmbg-perf] inference failed: %s\n",
                         aicore_rmbg_last_error(ctx)
                                 ? aicore_rmbg_last_error(ctx)
                                 : "unknown error");
            aicore_rmbg_free_buffer(alpha);
            aicore_rmbg_free(ctx);
            return 1;
        }

        aicore_rmbg_timings timing{};
        if (aicore_rmbg_last_timings(ctx, &timing) != 0 ||
            timing.preprocess_ms <= 0.0 || timing.inference_ms <= 0.0 ||
            timing.postprocess_ms <= 0.0 ||
            timing.total_ms + 0.01 < timing.preprocess_ms +
                                             timing.inference_ms +
                                             timing.postprocess_ms) {
            std::fprintf(stderr, "[rmbg-perf] invalid timing contract\n");
            aicore_rmbg_free_buffer(alpha);
            aicore_rmbg_free(ctx);
            return 1;
        }

        const uint64_t output_hash =
                fnv1a(alpha, static_cast<size_t>(width) * height);
        if (run == 0) {
            const auto alpha_range = std::minmax_element(
                    alpha, alpha + static_cast<size_t>(width) * height);
            if (*alpha_range.second - *alpha_range.first < 8) {
                std::fprintf(stderr,
                             "[rmbg-perf] degenerate alpha output: "
                             "range=[%u,%u]\n",
                             static_cast<unsigned>(*alpha_range.first),
                             static_cast<unsigned>(*alpha_range.second));
                aicore_rmbg_free_buffer(alpha);
                aicore_rmbg_free(ctx);
                return 1;
            }
        }
        aicore_rmbg_free_buffer(alpha);
        if (run == 0) {
            reference_hash = output_hash;
        } else if (run > 0 && output_hash != reference_hash) {
            std::fprintf(stderr,
                         "[rmbg-perf] nondeterministic output at run %d\n",
                         run);
            aicore_rmbg_free(ctx);
            return 1;
        }
        if (run >= 0) {
            samples.push_back(timing.inference_ms);
            std::printf("[rmbg-perf] run=%d graph_ms=%.3f total_ms=%.3f\n",
                        run + 1, timing.inference_ms, timing.total_ms);
        }
    }

    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const size_t p95_index =
            static_cast<size_t>(
                    std::ceil(0.95 * static_cast<double>(samples.size()))) -
            1;
    const double p95 = samples[p95_index];
    std::printf(
            "[rmbg-perf] device=%s median_ms=%.3f p95_ms=%.3f "
            "output_hash=%llu\n",
            device, median, p95,
            static_cast<unsigned long long>(reference_hash));

    const char* ceiling_env = std::getenv("AICORE_TEST_RMBG_MAX_MEDIAN_MS");
    const double ceiling = ceiling_env && ceiling_env[0]
                                   ? std::strtod(ceiling_env, nullptr)
                                   : 0.0;
    aicore_rmbg_free(ctx);
    if (ceiling > 0.0 && median > ceiling) {
        std::fprintf(stderr,
                     "[rmbg-perf] median %.3f ms exceeds %.3f ms ceiling\n",
                     median, ceiling);
        return 1;
    }
    return 0;
}
