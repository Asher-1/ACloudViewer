// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// ggml LoMa matcher bridge. It deliberately reuses the attention runtime but
// exposes LoMa's own feature contract, so a LoMa GGUF cannot be loaded through
// the public LightGlue ABI by accident.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/loma_capi.h"
#include "common/capi_utils.hpp"
#include "common/model_cache.hpp"
#include "tasks/lightglue/types.hpp"
#include "tasks/loma/descriptor.hpp"
#include "tasks/loma/detector.hpp"
#include "tasks/loma/quantize.hpp"

namespace {

aicore::lightglue::Features ToNative(const aicore_loma_features* source) {
    aicore::lightglue::Features result;
    if (source == nullptr) return result;
    result.descriptor_dim = source->descriptor_dim;
    result.image_width = source->image_width;
    result.image_height = source->image_height;
    if (source->keypoints != nullptr && source->n_keypoints > 0) {
        result.keypoints.resize(static_cast<size_t>(source->n_keypoints));
        for (int32_t index = 0; index < source->n_keypoints; ++index) {
            result.keypoints[static_cast<size_t>(index)].x =
                    source->keypoints[index].x;
            result.keypoints[static_cast<size_t>(index)].y =
                    source->keypoints[index].y;
        }
    }
    if (source->descriptors != nullptr && source->n_keypoints > 0 &&
        source->descriptor_dim > 0) {
        const size_t count = static_cast<size_t>(source->n_keypoints) *
                             static_cast<size_t>(source->descriptor_dim);
        result.descriptors.assign(source->descriptors,
                                  source->descriptors + count);
    }
    return result;
}

}  // namespace

struct aicore_loma_matcher_options {
    aicore::lightglue::MatchingOptions value;
    aicore_loma_matcher_options() {
        value.type = aicore::lightglue::FeatureMatcherType::kLoma;
        value.min_score = 0.1;
    }
};

struct aicore_loma_detector_options {
    aicore::loma::DetectorOptions value;
};

struct aicore_loma_detector_ctx {
    std::unique_ptr<aicore::loma::Detector> detector;
    std::string error;
    aicore_pipeline_timings timings{};
};

struct aicore_loma_descriptor_options {
    aicore::loma::DescriptorOptions value;
};

struct aicore_loma_descriptor_ctx {
    std::unique_ptr<aicore::loma::Descriptor> descriptor;
    std::string error;
    aicore_pipeline_timings timings{};
};

struct aicore_loma_matcher_ctx {
    std::unique_ptr<aicore::lightglue::FeatureMatcher> matcher;
    std::string error;
    aicore_pipeline_timings timings{};
};

extern "C" {

AICORE_CAPI int aicore_loma_abi_version(void) { return 4; }

AICORE_CAPI aicore_loma_detector_options* aicore_loma_detector_options_new(
        void) {
    return new (std::nothrow) aicore_loma_detector_options();
}

AICORE_CAPI void aicore_loma_detector_options_free(
        aicore_loma_detector_options* options) {
    delete options;
}

AICORE_CAPI void aicore_loma_detector_options_set_device(
        aicore_loma_detector_options* options, const char* device) {
    if (options != nullptr)
        options->value.device = device != nullptr ? device : "";
}

AICORE_CAPI void aicore_loma_detector_options_set_threads(
        aicore_loma_detector_options* options, int32_t threads) {
    if (options != nullptr) options->value.num_threads = threads;
}

AICORE_CAPI void aicore_loma_detector_options_set_max_keypoints(
        aicore_loma_detector_options* options, int32_t max_keypoints) {
    if (options != nullptr) options->value.max_keypoints = max_keypoints;
}

AICORE_CAPI aicore_loma_detector_ctx* aicore_loma_detector_load(
        const char* gguf_path, const aicore_loma_detector_options* options) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_loma_detector_ctx();
    if (ctx == nullptr) return nullptr;
    const aicore::loma::DetectorOptions native =
            options != nullptr ? options->value
                               : aicore_loma_detector_options().value;
    ctx->detector = std::make_unique<aicore::loma::Detector>();
    if (!ctx->detector->Load(gguf_path, native)) {
        ctx->error = ctx->detector->error();
        ctx->detector.reset();
    }
    return ctx;
}

AICORE_CAPI void aicore_loma_detector_free(aicore_loma_detector_ctx* ctx) {
    delete ctx;
}

AICORE_CAPI int aicore_loma_detector_is_ready(
        const aicore_loma_detector_ctx* ctx) {
    return ctx != nullptr && ctx->detector != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_loma_detector_last_error(
        const aicore_loma_detector_ctx* ctx) {
    if (ctx == nullptr) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

AICORE_CAPI int aicore_loma_detector_run(
        aicore_loma_detector_ctx* ctx,
        const aicore_loma_rgb_image* image,
        aicore_loma_detected_features* out_features) {
    const auto started = aicore::capi::PipelineClock::now();
    if (ctx == nullptr || ctx->detector == nullptr || image == nullptr ||
        out_features == nullptr) {
        return -1;
    }
    *out_features = {};
    aicore::loma::DetectorResult native;
    if (!ctx->detector->Detect(*image, &native)) {
        ctx->error = ctx->detector->error();
        return -1;
    }
    const size_t points_bytes =
            native.keypoints.size() * sizeof(aicore_loma_keypoint);
    const size_t scores_bytes = native.scores.size() * sizeof(float);
    auto* keypoints =
            static_cast<aicore_loma_keypoint*>(std::malloc(points_bytes));
    auto* scores = static_cast<float*>(std::malloc(scores_bytes));
    if ((keypoints == nullptr || scores == nullptr) &&
        !native.keypoints.empty()) {
        std::free(keypoints);
        std::free(scores);
        ctx->error = "failed to allocate DaD detector output";
        return -1;
    }
    if (!native.keypoints.empty()) {
        std::memcpy(keypoints, native.keypoints.data(), points_bytes);
        std::memcpy(scores, native.scores.data(), scores_bytes);
    }
    out_features->keypoints = keypoints;
    out_features->scores = scores;
    out_features->count = static_cast<int32_t>(native.keypoints.size());
    out_features->image_width = image->width;
    out_features->image_height = image->height;
    aicore::capi::record_pipeline_e2e(ctx->timings, started);
    return 0;
}

AICORE_CAPI void aicore_loma_detected_features_free(
        aicore_loma_detected_features* features) {
    if (features == nullptr) return;
    std::free(features->keypoints);
    std::free(features->scores);
    *features = {};
}

AICORE_CAPI aicore_loma_descriptor_options* aicore_loma_descriptor_options_new(
        void) {
    return new (std::nothrow) aicore_loma_descriptor_options();
}

AICORE_CAPI void aicore_loma_descriptor_options_free(
        aicore_loma_descriptor_options* options) {
    delete options;
}

AICORE_CAPI void aicore_loma_descriptor_options_set_device(
        aicore_loma_descriptor_options* options, const char* device) {
    if (options != nullptr)
        options->value.device = device != nullptr ? device : "";
}

AICORE_CAPI void aicore_loma_descriptor_options_set_threads(
        aicore_loma_descriptor_options* options, int32_t threads) {
    if (options != nullptr) options->value.num_threads = threads;
}

AICORE_CAPI aicore_loma_descriptor_ctx* aicore_loma_descriptor_load(
        const char* gguf_path, const aicore_loma_descriptor_options* options) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_loma_descriptor_ctx();
    if (ctx == nullptr) return nullptr;
    const aicore::loma::DescriptorOptions native =
            options != nullptr ? options->value
                               : aicore_loma_descriptor_options().value;
    ctx->descriptor = std::make_unique<aicore::loma::Descriptor>();
    if (!ctx->descriptor->Load(gguf_path, native)) {
        ctx->error = ctx->descriptor->error();
        ctx->descriptor.reset();
    }
    return ctx;
}

AICORE_CAPI void aicore_loma_descriptor_free(aicore_loma_descriptor_ctx* ctx) {
    delete ctx;
}

AICORE_CAPI int aicore_loma_descriptor_is_ready(
        const aicore_loma_descriptor_ctx* ctx) {
    return ctx != nullptr && ctx->descriptor != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_loma_descriptor_last_error(
        const aicore_loma_descriptor_ctx* ctx) {
    if (ctx == nullptr) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

AICORE_CAPI int aicore_loma_descriptor_run(
        aicore_loma_descriptor_ctx* ctx,
        const aicore_loma_rgb_image* image,
        const aicore_loma_keypoint* keypoints,
        int32_t count,
        int32_t keypoint_image_width,
        int32_t keypoint_image_height,
        aicore_loma_described_features* out_features) {
    const auto started = aicore::capi::PipelineClock::now();
    if (ctx == nullptr || ctx->descriptor == nullptr || image == nullptr ||
        keypoints == nullptr || count <= 0 || out_features == nullptr) {
        return -1;
    }
    *out_features = {};
    std::vector<float> native;
    if (!ctx->descriptor->Describe(*image, keypoints, count,
                                   keypoint_image_width, keypoint_image_height,
                                   &native)) {
        ctx->error = ctx->descriptor->error();
        return -1;
    }
    const size_t bytes = native.size() * sizeof(float);
    auto* descriptors = static_cast<float*>(std::malloc(bytes));
    if (descriptors == nullptr && !native.empty()) {
        ctx->error = "failed to allocate DeDoDe descriptor output";
        return -1;
    }
    if (!native.empty()) std::memcpy(descriptors, native.data(), bytes);
    out_features->descriptors = descriptors;
    out_features->count = count;
    out_features->descriptor_dim = ctx->descriptor->descriptor_dim();
    aicore::capi::record_pipeline_e2e(ctx->timings, started);
    return 0;
}

AICORE_CAPI void aicore_loma_described_features_free(
        aicore_loma_described_features* features) {
    if (features == nullptr) return;
    std::free(features->descriptors);
    *features = {};
}

AICORE_CAPI aicore_loma_matcher_options* aicore_loma_matcher_options_new(void) {
    return new (std::nothrow) aicore_loma_matcher_options();
}

AICORE_CAPI void aicore_loma_matcher_options_free(
        aicore_loma_matcher_options* options) {
    delete options;
}

AICORE_CAPI void aicore_loma_matcher_options_set_device(
        aicore_loma_matcher_options* options, const char* device) {
    if (options != nullptr)
        options->value.device = device != nullptr ? device : "";
}

AICORE_CAPI void aicore_loma_matcher_options_set_threads(
        aicore_loma_matcher_options* options, int32_t threads) {
    if (options != nullptr) options->value.num_threads = threads;
}

AICORE_CAPI void aicore_loma_matcher_options_set_min_score(
        aicore_loma_matcher_options* options, double min_score) {
    if (options != nullptr) options->value.min_score = min_score;
}

AICORE_CAPI aicore_loma_matcher_ctx* aicore_loma_matcher_load(
        const char* gguf_path, const aicore_loma_matcher_options* options) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_loma_matcher_ctx();
    if (ctx == nullptr) return nullptr;
    aicore::lightglue::MatchingOptions native =
            options != nullptr ? options->value
                               : aicore_loma_matcher_options().value;
    native.type = aicore::lightglue::FeatureMatcherType::kLoma;
    native.model_path = gguf_path;
    if (native.device.empty()) native.device = "auto";
    ctx->matcher =
            aicore::lightglue::create_feature_matcher(native, &ctx->error);
    return ctx;
}

AICORE_CAPI void aicore_loma_matcher_free(aicore_loma_matcher_ctx* ctx) {
    delete ctx;
}

AICORE_CAPI int aicore_loma_matcher_is_ready(
        const aicore_loma_matcher_ctx* ctx) {
    return ctx != nullptr && ctx->matcher != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_loma_matcher_last_error(
        const aicore_loma_matcher_ctx* ctx) {
    if (ctx == nullptr) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

AICORE_CAPI int aicore_loma_matcher_run(aicore_loma_matcher_ctx* ctx,
                                        const aicore_loma_features* image0,
                                        const aicore_loma_features* image1,
                                        aicore_loma_match** out_matches,
                                        int32_t* out_count) {
    const auto started = aicore::capi::PipelineClock::now();
    if (ctx == nullptr || ctx->matcher == nullptr || image0 == nullptr ||
        image1 == nullptr || out_matches == nullptr || out_count == nullptr) {
        return -1;
    }
    *out_matches = nullptr;
    *out_count = 0;
    std::vector<aicore::lightglue::Match> matches;
    if (!ctx->matcher->match(ToNative(image0), ToNative(image1), &matches)) {
        ctx->error = ctx->matcher->error();
        return -1;
    }
    if (!matches.empty()) {
        auto* output = static_cast<aicore_loma_match*>(
                std::malloc(matches.size() * sizeof(aicore_loma_match)));
        if (output == nullptr) return -1;
        for (size_t index = 0; index < matches.size(); ++index) {
            output[index] = {matches[index].point2D_idx1,
                             matches[index].point2D_idx2, matches[index].score};
        }
        *out_matches = output;
        *out_count = static_cast<int32_t>(matches.size());
    }
    aicore::capi::record_pipeline_e2e(ctx->timings, started);
    return 0;
}

AICORE_CAPI void aicore_loma_free_matches(aicore_loma_match* matches) {
    std::free(matches);
}

AICORE_CAPI int aicore_loma_matcher_last_pipeline_timings(
        const aicore_loma_matcher_ctx* ctx, aicore_pipeline_timings* out) {
    return ctx != nullptr
                   ? aicore::capi::copy_pipeline_timings(ctx->timings, out)
                   : -1;
}

AICORE_CAPI int aicore_loma_warmup_backend(const char* device) {
    return aicore_warmup_backend(device);
}

AICORE_CAPI int aicore_loma_quantize_gguf(const char* input_gguf,
                                          const char* output_gguf,
                                          const char* type) {
    if (input_gguf == nullptr || output_gguf == nullptr || type == nullptr)
        return -1;
    std::string error;
    if (aicore::loma::QuantizeModel(input_gguf, output_gguf, type, &error))
        return 0;
    std::fprintf(stderr, "LoMa GGUF quantization failed: %s\n", error.c_str());
    return -1;
}

AICORE_CAPI char* aicore_loma_model_cache_dir(void) {
    return aicore::capi::dup_cstr(aicore::loma_model_cache_dir());
}

}  // extern "C"
