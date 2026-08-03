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

#include "aicore/backend_capi.h"
#include "aicore/deeplsd_capi.h"
#include "deeplsd.hpp"
#include "ggml_backend_utils.hpp"
#include "gguf_weight_quantize.hpp"
#include "model_cache.hpp"

namespace {

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

std::string normalize_device(const char* device) {
    if (device == nullptr || device[0] == '\0') {
        return "cpu";
    }
    return ggml_common::resolve_device_request(device);
}

}  // namespace

struct aicore_deeplsd_options {
    std::string device = "cpu";
    int32_t threads = 0;
};

struct aicore_deeplsd_ctx {
    std::unique_ptr<deeplsd::DeepLSDExtractor> extractor;
    std::string model_path;
    std::string device;
    std::string last_error;
};

AICORE_CAPI int aicore_deeplsd_abi_version(void) { return 1; }

AICORE_CAPI aicore_deeplsd_options* aicore_deeplsd_options_new(void) {
    return new aicore_deeplsd_options();
}

AICORE_CAPI void aicore_deeplsd_options_free(aicore_deeplsd_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_deeplsd_options_set_device(aicore_deeplsd_options* opts,
                                                   const char* device) {
    if (opts != nullptr && device != nullptr) {
        opts->device = device;
    }
}

AICORE_CAPI void aicore_deeplsd_options_set_threads(
        aicore_deeplsd_options* opts, int n_threads) {
    if (opts != nullptr) {
        opts->threads = n_threads;
    }
}

AICORE_CAPI aicore_deeplsd_ctx* aicore_deeplsd_load_opts(
        const char* gguf_path, const aicore_deeplsd_options* opts) {
    if (gguf_path == nullptr) {
        return nullptr;
    }
    auto* ctx = new (std::nothrow) aicore_deeplsd_ctx();
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->model_path = gguf_path;
    ctx->device =
            opts != nullptr ? normalize_device(opts->device.c_str()) : "cpu";

    deeplsd::DeepLSDOptions o;
    o.model_path = ctx->model_path;
    o.device = ctx->device;
    o.num_threads = opts != nullptr ? opts->threads : 4;
    o.use_ggml_cnn = true;

    std::string err;
    ctx->extractor = deeplsd::CreateDeepLSDExtractor(o, &err);
    if (!ctx->extractor) {
        ctx->last_error =
                err.empty() ? "failed to create DeepLSD extractor" : err;
    } else if (!ctx->extractor->Error().empty()) {
        ctx->last_error = ctx->extractor->Error();
    }
    return ctx;
}

AICORE_CAPI void aicore_deeplsd_free(aicore_deeplsd_ctx* ctx) { delete ctx; }

AICORE_CAPI const char* aicore_deeplsd_last_error(
        const aicore_deeplsd_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI int aicore_deeplsd_extract_gray(aicore_deeplsd_ctx* ctx,
                                            const uint8_t* gray,
                                            int32_t width,
                                            int32_t height,
                                            int32_t row_stride,
                                            float** out_distance,
                                            float** out_angle,
                                            int32_t* out_width,
                                            int32_t* out_height) {
    if (ctx == nullptr || ctx->extractor == nullptr || gray == nullptr ||
        out_distance == nullptr || out_angle == nullptr ||
        out_width == nullptr || out_height == nullptr || width <= 0 ||
        height <= 0) {
        return -1;
    }
    *out_distance = nullptr;
    *out_angle = nullptr;

    deeplsd::DeepLSDResult result;
    if (!ctx->extractor->ExtractFromGray(gray, width, height, row_stride,
                                         &result)) {
        ctx->last_error = ctx->extractor->Error();
        return -1;
    }

    const size_t plane = static_cast<size_t>(result.width) * result.height;
    auto* df = static_cast<float*>(std::malloc(plane * sizeof(float)));
    auto* ang = static_cast<float*>(std::malloc(plane * sizeof(float)));
    if (df == nullptr || ang == nullptr) {
        std::free(df);
        std::free(ang);
        ctx->last_error = "out of memory";
        return -1;
    }
    std::memcpy(df, result.distance_field.data(), plane * sizeof(float));
    std::memcpy(ang, result.angle_field.data(), plane * sizeof(float));
    *out_distance = df;
    *out_angle = ang;
    *out_width = result.width;
    *out_height = result.height;
    return 0;
}

AICORE_CAPI int aicore_deeplsd_extract_segments(
        aicore_deeplsd_ctx* ctx,
        const uint8_t* gray,
        int32_t width,
        int32_t height,
        int32_t row_stride,
        aicore_deeplsd_segment** out_segments,
        int32_t* out_segment_count,
        float** out_distance,
        float** out_angle,
        int32_t* out_width,
        int32_t* out_height) {
    if (ctx == nullptr || ctx->extractor == nullptr || gray == nullptr ||
        out_distance == nullptr || out_angle == nullptr ||
        out_width == nullptr || out_height == nullptr || width <= 0 ||
        height <= 0) {
        return -1;
    }
    *out_distance = nullptr;
    *out_angle = nullptr;
    if (out_segments != nullptr) {
        *out_segments = nullptr;
    }
    if (out_segment_count != nullptr) {
        *out_segment_count = 0;
    }

    deeplsd::DeepLSDResult result;
    if (!ctx->extractor->ExtractFromGray(gray, width, height, row_stride,
                                         &result)) {
        ctx->last_error = ctx->extractor->Error();
        return -1;
    }

    const size_t plane = static_cast<size_t>(result.width) * result.height;
    auto* df = static_cast<float*>(std::malloc(plane * sizeof(float)));
    auto* ang = static_cast<float*>(std::malloc(plane * sizeof(float)));
    if (df == nullptr || ang == nullptr) {
        std::free(df);
        std::free(ang);
        ctx->last_error = "out of memory";
        return -1;
    }
    std::memcpy(df, result.distance_field.data(), plane * sizeof(float));
    std::memcpy(ang, result.angle_field.data(), plane * sizeof(float));
    *out_distance = df;
    *out_angle = ang;
    *out_width = result.width;
    *out_height = result.height;

    if (out_segments == nullptr || out_segment_count == nullptr) {
        return 0;
    }

    const int32_t count = static_cast<int32_t>(result.segments.size());
    if (count == 0) {
        return 0;
    }

    auto* segs = static_cast<aicore_deeplsd_segment*>(
            std::malloc(static_cast<size_t>(count) *
                        sizeof(aicore_deeplsd_segment)));
    if (segs == nullptr) {
        ctx->last_error = "out of memory";
        return -1;
    }
    for (int32_t i = 0; i < count; ++i) {
        segs[i].x1 = result.segments[static_cast<size_t>(i)].x1;
        segs[i].y1 = result.segments[static_cast<size_t>(i)].y1;
        segs[i].x2 = result.segments[static_cast<size_t>(i)].x2;
        segs[i].y2 = result.segments[static_cast<size_t>(i)].y2;
        segs[i].score = result.segments[static_cast<size_t>(i)].score;
    }
    *out_segments = segs;
    *out_segment_count = count;
    return 0;
}

AICORE_CAPI char* aicore_deeplsd_info_json(aicore_deeplsd_ctx* ctx) {
    if (ctx == nullptr || ctx->extractor == nullptr) {
        return dup_cstr("{}");
    }
    const std::string json =
            std::string(
                    "{\n  \"architecture\": \"deeplsd\",\n  \"device\": \"") +
            ctx->extractor->Device() + "\",\n  \"model\": \"" +
            ctx->model_path + "\"\n}";
    return dup_cstr(json);
}

AICORE_CAPI void aicore_deeplsd_free_string(char* s) { std::free(s); }

AICORE_CAPI int aicore_deeplsd_warmup_backend(const char* device) {
    return aicore_warmup_backend(device);
}

AICORE_CAPI char* aicore_deeplsd_model_cache_dir(void) {
    return dup_cstr(aicore::deeplsd_model_cache_dir());
}

AICORE_CAPI int aicore_deeplsd_quantize(const char* input_gguf,
                                        const char* output_gguf,
                                        const char* type) {
    if (input_gguf == nullptr || output_gguf == nullptr || type == nullptr) {
        return -1;
    }
    std::string err;
    if (!aicore::common::quantize_gguf_weights(input_gguf, output_gguf, type,
                                               &err)) {
        return -1;
    }
    return 0;
}
