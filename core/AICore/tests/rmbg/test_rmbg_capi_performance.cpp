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

// The math profile must travel as options; no rmbg call may write ggml-side
// environment variables (the former RMBG_VK_* bridge leaked into unrelated
// tasks in the same process, e.g. qTrellis DINO on coopmat2).
const char* const kProfileEnvKeys[] = {
        "RMBG_VK_COOPMAT_MATMUL", "RMBG_VK_SCALAR_DIRECT_CONV",
        "RMBG_CUDA_CONV_TF32", "GGML_VK_DISABLE_F16"};

std::string profile_env_snapshot() {
    std::string snapshot;
    for (const char* key : kProfileEnvKeys) {
        const char* value = std::getenv(key);
        snapshot += key;
        snapshot += value ? std::string("=") + value : "=<unset>";
        snapshot += ";";
    }
    return snapshot;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string env_before = profile_env_snapshot();
    // Fails the probe when the rmbg C API wrote any ggml-side env var.
    auto check_env_untouched = [&env_before]() {
        if (profile_env_snapshot() != env_before) {
            std::fprintf(stderr,
                         "[rmbg-perf] ggml-side env vars were written by the "
                         "rmbg C API (the math profile must travel as "
                         "options, never as environment variables)\n");
            return false;
        }
        return true;
    };

    const char* model = std::getenv("AICORE_TEST_RMBG_GGUF");
    if (!model || !model[0]) {
        std::printf("[rmbg-perf] skipped: AICORE_TEST_RMBG_GGUF is unset\n");
        return 77;
    }
    const char* device = argc > 1 ? argv[1]
                                  : env_or("AICORE_TEST_RMBG_DEVICE",
                                           "AICORE_TEST_DEVICE", "auto");
    // Optional math profile override (argv[2]); default keeps the task's
    // own profile resolution ("optimized").
    const char* profile = argc > 2 ? argv[2] : nullptr;
    if (std::strcmp(device, "cpu") == 0) {
        std::printf(
                "[rmbg-perf] skipped: GPU performance test requested CPU\n");
        return 77;
    }

    aicore_rmbg_options* options = aicore_rmbg_options_new();
    if (!options) return 1;
    aicore_rmbg_options_set_device(options, device);
    if (profile && profile[0]) {
        aicore_rmbg_options_set_math_profile(options, profile);
    }
    aicore_rmbg_ctx* ctx = aicore_rmbg_load_opts(model, options);
    aicore_rmbg_options_free(options);
    if (!ctx || !aicore_rmbg_is_ready(ctx)) {
        std::printf("[rmbg-perf] skipped: %s\n",
                    ctx && aicore_rmbg_last_error(ctx)
                            ? aicore_rmbg_last_error(ctx)
                            : "backend or model unavailable");
        aicore_rmbg_free(ctx);
        const bool env_ok = check_env_untouched();
        return env_ok ? 77 : 1;
    }

    char* info = aicore_rmbg_info_json(ctx);
    if (!info || !std::strstr(info, "\"math_profile\":")) {
        std::fprintf(stderr, "[rmbg-perf] missing math_profile in info JSON\n");
        aicore_rmbg_free_buffer(info);
        aicore_rmbg_free(ctx);
        return 1;
    }
    std::printf("[rmbg-perf] info=%s\n", info);
    aicore_rmbg_free_buffer(info);

    std::vector<uint8_t> rgb((size_t)kInputSize * kInputSize * 3);
    constexpr size_t kViewStride = static_cast<size_t>(kInputSize) * 4 + 16;
    std::vector<uint8_t> bgra(kViewStride * kInputSize, 0xA5);
    for (int y = 0; y < kInputSize; ++y) {
        for (int x = 0; x < kInputSize; ++x) {
            const size_t p = ((size_t)y * kInputSize + x) * 3;
            rgb[p + 0] = static_cast<uint8_t>((x * 13 + y * 3) & 255);
            rgb[p + 1] = static_cast<uint8_t>((x * 5 + y * 11) & 255);
            rgb[p + 2] = static_cast<uint8_t>((x ^ y) & 255);
            uint8_t* pixel = bgra.data() +
                             static_cast<size_t>(y) * kViewStride +
                             static_cast<size_t>(x) * 4;
            pixel[0] = rgb[p + 2];
            pixel[1] = rgb[p + 1];
            pixel[2] = rgb[p + 0];
            pixel[3] = 255;
        }
    }
    const aicore_image_view image{bgra.data(), kInputSize, kInputSize,
                                  kViewStride, AICORE_IMAGE_BGRA8};

    // One compatibility check outside the benchmark window: the legacy tight
    // RGB wrapper and a padded native BGRA view must produce identical mattes.
    uint8_t* legacy_alpha = nullptr;
    uint8_t* view_alpha = nullptr;
    int32_t legacy_w = 0, legacy_h = 0, view_w = 0, view_h = 0;
    const int legacy_rc =
            aicore_rmbg_alpha_mat_rgb(ctx, rgb.data(), kInputSize, kInputSize,
                                      &legacy_alpha, &legacy_w, &legacy_h);
    const int view_rc = aicore_rmbg_alpha_mat_image_view(
            ctx, &image, &view_alpha, &view_w, &view_h);
    const size_t matte_bytes = static_cast<size_t>(kInputSize) * kInputSize;
    const bool views_match =
            legacy_rc == 0 && view_rc == 0 && legacy_alpha && view_alpha &&
            legacy_w == view_w && legacy_h == view_h &&
            legacy_w == kInputSize && legacy_h == kInputSize &&
            std::memcmp(legacy_alpha, view_alpha, matte_bytes) == 0;
    aicore_rmbg_free_buffer(legacy_alpha);
    aicore_rmbg_free_buffer(view_alpha);
    if (!views_match) {
        std::fprintf(stderr, "[rmbg-perf] RGB/BGRA image-view parity failed\n");
        aicore_rmbg_free(ctx);
        return 1;
    }

    std::vector<double> samples;
    samples.reserve(kRuns);
    uint64_t reference_hash = 0;
    for (int run = -kWarmups; run < kRuns; ++run) {
        uint8_t* alpha = nullptr;
        int32_t width = 0;
        int32_t height = 0;
        if (aicore_rmbg_alpha_mat_image_view(ctx, &image, &alpha, &width,
                                             &height) != 0 ||
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
            "[rmbg-perf] device=%s profile=%s median_ms=%.3f p95_ms=%.3f "
            "output_hash=%llu\n",
            device, profile && profile[0] ? profile : "default", median, p95,
            static_cast<unsigned long long>(reference_hash));

    const char* ceiling_env = std::getenv("AICORE_TEST_RMBG_MAX_MEDIAN_MS");
    const double ceiling = ceiling_env && ceiling_env[0]
                                   ? std::strtod(ceiling_env, nullptr)
                                   : 0.0;
    aicore_rmbg_free(ctx);
    if (!check_env_untouched()) {
        return 1;
    }
    if (ceiling > 0.0 && median > ceiling) {
        std::fprintf(stderr,
                     "[rmbg-perf] median %.3f ms exceeds %.3f ms ceiling\n",
                     median, ceiling);
        return 1;
    }
    return 0;
}
