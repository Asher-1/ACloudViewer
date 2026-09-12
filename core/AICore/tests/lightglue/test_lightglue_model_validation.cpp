// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "aicore/lightglue_capi.h"
#include "tests/common/validation_probe.hpp"

namespace {

double percentile(std::vector<double> values, double q) {
    std::sort(values.begin(), values.end());
    const size_t index = static_cast<size_t>(std::ceil(q * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

struct OwnedFeatures {
    std::vector<aicore_lightglue_keypoint> keypoints;
    std::vector<float> descriptors;
    aicore_lightglue_features view{};
};

OwnedFeatures makeFeatures(const aicore_lightglue_geometry& geometry,
                           int count) {
    OwnedFeatures result;
    result.keypoints.resize(count);
    result.descriptors.resize(static_cast<size_t>(count) * geometry.input_dim);
    for (int i = 0; i < count; ++i) {
        auto& keypoint = result.keypoints[i];
        keypoint.x = 24.0f + static_cast<float>((i * 67) % 592);
        keypoint.y = 18.0f + static_cast<float>((i * 43) % 444);
        keypoint.scale = 1.0f + 0.01f * static_cast<float>(i % 11);
        keypoint.orientation = 0.05f * static_cast<float>(i % 13);
        double norm = 0.0;
        float* descriptor = result.descriptors.data() +
                            static_cast<size_t>(i) * geometry.input_dim;
        for (int j = 0; j < geometry.input_dim; ++j) {
            const float value =
                    std::sin(static_cast<float>((i + 1) * (j + 3)) * 0.017f) +
                    std::cos(static_cast<float>((i + 7) * (j + 1)) * 0.011f);
            descriptor[j] = value;
            norm += static_cast<double>(value) * value;
        }
        const float inverse_norm =
                1.0f / static_cast<float>(std::sqrt(std::max(norm, 1e-20)));
        for (int j = 0; j < geometry.input_dim; ++j) {
            descriptor[j] *= inverse_norm;
        }
    }
    result.view.keypoints = result.keypoints.data();
    result.view.n_keypoints = count;
    result.view.descriptors = result.descriptors.data();
    result.view.descriptor_dim = geometry.input_dim;
    result.view.image_width = 640;
    result.view.image_height = 480;
    return result;
}

uint64_t hashMatches(const aicore_lightglue_match* matches, int count) {
    uint64_t hash = 1469598103934665603ULL;
    for (int i = 0; i < count; ++i) {
        hash = aicore::test::fnv1aAppend(hash, &matches[i].idx1,
                                         sizeof(matches[i].idx1));
        hash = aicore::test::fnv1aAppend(hash, &matches[i].idx2,
                                         sizeof(matches[i].idx2));
        hash = aicore::test::fnv1aAppend(hash, &matches[i].score,
                                         sizeof(matches[i].score));
    }
    return hash;
}

}  // namespace

int main(int argc, char** argv) {
    char* cache_dir_buf = aicore_lightglue_model_cache_dir();
    const std::string cache_dir = cache_dir_buf ? cache_dir_buf : "";
    aicore_lightglue_free_buffer(cache_dir_buf);
    // Bare `ctest` runs pass no args: resolve the published-catalog default
    // matcher so the probe skips (77) without local assets instead of
    // failing on a usage error. Explicit args keep the manual contract.
    const std::string matcher =
            argc >= 2 ? std::string(argv[1])
                      : cache_dir + "/aliked-lightglue-f16.gguf";
    const std::string device = argc >= 3 ? std::string(argv[2]) : "auto";
    const int warmups = argc >= 4 ? std::max(0, std::atoi(argv[3])) : 2;
    const int runs = argc >= 5 ? std::max(1, std::atoi(argv[4])) : 10;
    if (std::FILE* probe = std::fopen(matcher.c_str(), "rb")) {
        std::fclose(probe);
    } else {
        std::fprintf(stderr,
                     "[lightglue-validation] skipped: matcher model not "
                     "found: %s\n",
                     matcher.c_str());
        return 77;
    }

    aicore_lightglue_options* options = aicore_lightglue_options_new();
    aicore_lightglue_options_set_device(options, device.c_str());
    aicore_lightglue_options_set_matcher_type(options, 0);
    aicore_lightglue_options_set_min_score(options, 0.0);
    aicore_lightglue_ctx* ctx =
            aicore_lightglue_load_opts(matcher.c_str(), options);
    aicore_lightglue_options_free(options);
    aicore_lightglue_geometry geometry{};
    if (!aicore_lightglue_is_ready(ctx) ||
        aicore_lightglue_geometry_of(ctx, &geometry) != 0 ||
        geometry.input_dim <= 0) {
        std::fprintf(stderr, "LightGlue load failed: %s\n",
                     aicore_lightglue_last_error(ctx));
        aicore_lightglue_free(ctx);
        return 1;
    }

    OwnedFeatures features = makeFeatures(geometry, 64);
    std::vector<double> timings;
    uint64_t reference_hash = 0;
    int reference_count = -1;
    for (int run = -warmups; run < runs; ++run) {
        aicore_lightglue_match* matches = nullptr;
        int32_t count = 0;
        const auto started = std::chrono::steady_clock::now();
        const int rc = aicore_lightglue_run_match(
                ctx, &features.view, &features.view, &matches, &count);
        const auto stopped = std::chrono::steady_clock::now();
        if (rc != 0 || count <= 0) {
            std::fprintf(stderr,
                         "LightGlue inference failed: count=%d error=%s\n",
                         count, aicore_lightglue_last_error(ctx));
            aicore_lightglue_free_matches(matches);
            aicore_lightglue_free(ctx);
            return 1;
        }
        for (int i = 0; i < count; ++i) {
            if (matches[i].idx1 < 0 || matches[i].idx1 >= 64 ||
                matches[i].idx2 < 0 || matches[i].idx2 >= 64 ||
                !std::isfinite(matches[i].score)) {
                std::fprintf(stderr, "LightGlue returned an invalid match\n");
                aicore_lightglue_free_matches(matches);
                aicore_lightglue_free(ctx);
                return 1;
            }
        }
        const uint64_t hash = hashMatches(matches, count);
        if (run >= 0) {
            if (reference_count >= 0 &&
                (reference_count != count || reference_hash != hash)) {
                std::fprintf(
                        stderr,
                        "LightGlue output changed across runs: count=%d/%d\n",
                        reference_count, count);
                aicore_lightglue_free_matches(matches);
                aicore_lightglue_free(ctx);
                return 1;
            }
            reference_count = count;
            reference_hash = hash;
            timings.push_back(
                    std::chrono::duration<double, std::milli>(stopped - started)
                            .count());
        }
        aicore_lightglue_free_matches(matches);
    }
    std::printf(
            "{\"suite\":\"aicore-validation\",\"task\":\"lightglue\","
            "\"device\":\"%s\",\"feature_type\":%d,\"matches\":%d,"
            "\"inference_p50_ms\":%.6f,\"inference_p95_ms\":%.6f,"
            "\"output_hash\":\"%016llx\"}\n",
            device.c_str(), geometry.feature_type, reference_count,
            percentile(timings, 0.5), percentile(timings, 0.95),
            static_cast<unsigned long long>(reference_hash));
    aicore_lightglue_free(ctx);
    return 0;
}
