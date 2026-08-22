// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "aicore/aliked_capi.h"
#include "common/capi_utils.hpp"
#include "common/model_cache.hpp"
#include "tasks/aliked/include/lightglue/aliked.h"

namespace {

enum class FeatureState { kValid, kEmpty, kInvalid };

FeatureState feature_state(const lightglue::Features& features) {
    const size_t count = features.keypoints.size();
    if (count == 0) {
        return features.descriptors.empty() ? FeatureState::kEmpty
                                            : FeatureState::kInvalid;
    }
    if (features.descriptor_dim <= 0) {
        return FeatureState::kInvalid;
    }
    const size_t dim = static_cast<size_t>(features.descriptor_dim);
    if (count > std::numeric_limits<size_t>::max() / dim ||
        features.descriptors.size() != count * dim) {
        return FeatureState::kInvalid;
    }
    for (const auto& keypoint : features.keypoints) {
        if (!std::isfinite(keypoint.x) || !std::isfinite(keypoint.y) ||
            !std::isfinite(keypoint.scale) ||
            !std::isfinite(keypoint.orientation)) {
            return FeatureState::kInvalid;
        }
    }
    for (size_t i = 0; i < count; ++i) {
        double norm_squared = 0.0;
        for (size_t j = 0; j < dim; ++j) {
            const float value = features.descriptors[i * dim + j];
            if (!std::isfinite(value)) {
                return FeatureState::kInvalid;
            }
            norm_squared += static_cast<double>(value) * value;
        }
        if (!(norm_squared > 0.0) || !std::isfinite(norm_squared)) {
            return FeatureState::kInvalid;
        }
    }
    return FeatureState::kValid;
}

bool fill_features(const lightglue::Features& src,
                   aicore_lightglue_features* dst) {
    if (dst == nullptr ||
        src.keypoints.size() >
                static_cast<size_t>(std::numeric_limits<int32_t>::max()) ||
        src.descriptor_dim < 0) {
        return false;
    }
    const size_t count = src.keypoints.size();
    const size_t dim = static_cast<size_t>(src.descriptor_dim);
    if (dim != 0 && count > std::numeric_limits<size_t>::max() / dim) {
        return false;
    }
    const size_t descriptor_count = count * dim;
    if (src.descriptors.size() != descriptor_count) {
        return false;
    }

    *dst = {};
    dst->n_keypoints = static_cast<int32_t>(src.keypoints.size());
    dst->descriptor_dim = src.descriptor_dim;
    dst->image_width = src.image_width;
    dst->image_height = src.image_height;
    if (count == 0) {
        return true;
    }
    dst->keypoints = static_cast<aicore_lightglue_keypoint*>(
            std::malloc(sizeof(aicore_lightglue_keypoint) * count));
    dst->descriptors = descriptor_count == 0
                               ? nullptr
                               : static_cast<float*>(std::malloc(
                                         sizeof(float) * descriptor_count));
    if (dst->keypoints == nullptr ||
        (descriptor_count != 0 && dst->descriptors == nullptr)) {
        std::free(dst->keypoints);
        std::free(dst->descriptors);
        *dst = {};
        return false;
    }
    for (int32_t i = 0; i < dst->n_keypoints; ++i) {
        dst->keypoints[i].x = src.keypoints[static_cast<size_t>(i)].x;
        dst->keypoints[i].y = src.keypoints[static_cast<size_t>(i)].y;
        dst->keypoints[i].scale = src.keypoints[static_cast<size_t>(i)].scale;
        dst->keypoints[i].orientation =
                src.keypoints[static_cast<size_t>(i)].orientation;
    }
    if (descriptor_count != 0) {
        std::memcpy(dst->descriptors, src.descriptors.data(),
                    descriptor_count * sizeof(float));
    }
    return true;
}

}  // namespace

using aicore::capi::dup_cstr;

struct aicore_aliked_options {
    std::string device = "cpu";
    int32_t threads = 0;
    int32_t max_keypoints = 1024;
    int32_t resize_long_edge = 1024;
};

struct aicore_aliked_ctx {
    std::unique_ptr<lightglue::AlikedFeatureExtractor> extractor;
    std::string model_path;
    std::string device;
    std::string last_error;
};

AICORE_CAPI int aicore_aliked_abi_version(void) { return 2; }

AICORE_CAPI aicore_aliked_options* aicore_aliked_options_new(void) {
    return new aicore_aliked_options();
}

AICORE_CAPI void aicore_aliked_options_free(aicore_aliked_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_aliked_options_set_device(aicore_aliked_options* opts,
                                                  const char* device) {
    if (opts != nullptr && device != nullptr) {
        opts->device = device;
    }
}

AICORE_CAPI void aicore_aliked_options_set_threads(aicore_aliked_options* opts,
                                                   int n_threads) {
    if (opts != nullptr) {
        opts->threads = n_threads;
    }
}

AICORE_CAPI void aicore_aliked_options_set_max_keypoints(
        aicore_aliked_options* opts, int32_t max_keypoints) {
    if (opts != nullptr) {
        opts->max_keypoints = max_keypoints;
    }
}

AICORE_CAPI void aicore_aliked_options_set_resize_long_edge(
        aicore_aliked_options* opts, int32_t px) {
    if (opts != nullptr) {
        opts->resize_long_edge = px;
    }
}

AICORE_CAPI aicore_aliked_ctx* aicore_aliked_load_opts(
        const char* gguf_path, const aicore_aliked_options* opts) {
    if (gguf_path == nullptr) {
        return nullptr;
    }
    auto* ctx = new (std::nothrow) aicore_aliked_ctx();
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->model_path = gguf_path;
    lightglue::AlikedExtractionOptions lg_opts;
    lg_opts.model_path = gguf_path;
    lg_opts.use_ggml_cnn = true;
    if (opts != nullptr) {
        lg_opts.device = opts->device;
        lg_opts.num_threads = opts->threads;
        lg_opts.max_keypoints = opts->max_keypoints;
        lg_opts.resize_long_edge = opts->resize_long_edge;
        ctx->device = opts->device;
    }
    std::string err;
    ctx->extractor = lightglue::CreateAlikedFeatureExtractor(lg_opts, &err);
    if (!ctx->extractor) {
        ctx->last_error = err.empty() ? "failed to load ALIKED model" : err;
        return ctx;
    }
    ctx->device = ctx->extractor->Device();
    return ctx;
}

AICORE_CAPI void aicore_aliked_free(aicore_aliked_ctx* ctx) { delete ctx; }

AICORE_CAPI int aicore_aliked_is_ready(const aicore_aliked_ctx* ctx) {
    return ctx != nullptr && ctx->extractor != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_aliked_last_error(const aicore_aliked_ctx* ctx) {
    if (ctx == nullptr) {
        return "null context";
    }
    if (!ctx->last_error.empty()) {
        return ctx->last_error.c_str();
    }
    if (ctx->extractor) {
        return ctx->extractor->Error().c_str();
    }
    return "";
}

AICORE_CAPI int aicore_aliked_extract_rgb(aicore_aliked_ctx* ctx,
                                          const uint8_t* rgb,
                                          int32_t width,
                                          int32_t height,
                                          int32_t row_stride,
                                          aicore_lightglue_features* out) {
    if (out != nullptr) {
        *out = {};
    }
    if (ctx == nullptr || !ctx->extractor || rgb == nullptr || out == nullptr) {
        if (ctx) ctx->last_error = "invalid extract arguments";
        return -1;
    }
    lightglue::Features features;
    if (!ctx->extractor->ExtractFromRgb(rgb, width, height, row_stride,
                                        &features)) {
        ctx->last_error = ctx->extractor->Error();
        return -1;
    }
    FeatureState state = feature_state(features);
    if (ctx->device.find("Vulkan") != std::string::npos &&
        state != FeatureState::kValid) {
        // A freshly created Vulkan session can report a successful first
        // submission before its pipelines have produced usable output. The
        // extractor scopes transient graphs per call, so a second call is a
        // clean submission using the already initialized pipelines.
        lightglue::Features retry;
        if (!ctx->extractor->ExtractFromRgb(rgb, width, height, row_stride,
                                            &retry)) {
            ctx->last_error = ctx->extractor->Error();
            return -1;
        }
        const FeatureState retry_state = feature_state(retry);
        if (retry_state == FeatureState::kInvalid) {
            ctx->last_error =
                    "ALIKED Vulkan retry produced invalid feature data";
            return -1;
        }
        features = std::move(retry);
        state = retry_state;
    }
    if (state == FeatureState::kInvalid) {
        ctx->last_error = "ALIKED extractor produced invalid feature data";
        return -1;
    }
    if (!fill_features(features, out)) {
        ctx->last_error = "ALIKED feature output allocation or shape failure";
        return -1;
    }
    ctx->last_error.clear();
    return 0;
}

AICORE_CAPI char* aicore_aliked_info_json(aicore_aliked_ctx* ctx) {
    if (ctx == nullptr || !ctx->extractor) {
        return dup_cstr("{}");
    }
    const std::string json = std::string("{\"device\":\"") +
                             ctx->extractor->Device() + "\",\"model\":\"" +
                             ctx->model_path + "\"}";
    return dup_cstr(json);
}

AICORE_CAPI char* aicore_aliked_model_cache_dir(void) {
    return dup_cstr(aicore::aliked_model_cache_dir());
}

AICORE_CAPI int aicore_aliked_quantize_gguf(const char* input_gguf,
                                            const char* output_gguf,
                                            const char* type) {
    if (input_gguf == nullptr || output_gguf == nullptr || type == nullptr) {
        return -1;
    }
    std::string error;
    return lightglue::QuantizeAlikedModel(input_gguf, output_gguf, type, &error)
                   ? 0
                   : -1;
}

AICORE_CAPI void aicore_aliked_free_buffer(void* p) { std::free(p); }
