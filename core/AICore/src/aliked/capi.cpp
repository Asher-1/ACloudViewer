// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>

#include "aicore/aliked_capi.h"
#include "lightglue/aliked.h"
#include "model_cache.hpp"

namespace {

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

void fill_features(const lightglue::Features& src,
                   aicore_lightglue_features* dst) {
    if (dst == nullptr) {
        return;
    }
    dst->n_keypoints = static_cast<int32_t>(src.keypoints.size());
    dst->descriptor_dim = src.descriptor_dim;
    dst->image_width = src.image_width;
    dst->image_height = src.image_height;
    dst->keypoints = static_cast<aicore_lightglue_keypoint*>(
            std::malloc(sizeof(aicore_lightglue_keypoint) *
                        static_cast<size_t>(dst->n_keypoints)));
    dst->descriptors = static_cast<float*>(
            std::malloc(sizeof(float) * static_cast<size_t>(dst->n_keypoints) *
                        static_cast<size_t>(std::max(1, dst->descriptor_dim))));
    for (int32_t i = 0; i < dst->n_keypoints; ++i) {
        dst->keypoints[i].x = src.keypoints[static_cast<size_t>(i)].x;
        dst->keypoints[i].y = src.keypoints[static_cast<size_t>(i)].y;
        dst->keypoints[i].scale = src.keypoints[static_cast<size_t>(i)].scale;
        dst->keypoints[i].orientation =
                src.keypoints[static_cast<size_t>(i)].orientation;
    }
    if (dst->descriptors && !src.descriptors.empty()) {
        std::memcpy(dst->descriptors, src.descriptors.data(),
                    src.descriptors.size() * sizeof(float));
    }
}

}  // namespace

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

AICORE_CAPI int aicore_aliked_abi_version(void) { return 1; }

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
    fill_features(features, out);
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

AICORE_CAPI void aicore_aliked_free_string(char* s) { std::free(s); }
