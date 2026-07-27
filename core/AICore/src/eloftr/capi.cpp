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
#include <vector>

#include "aicore/eloftr_capi.h"
#include "eloftr.hpp"
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

const char* resolve_device(const char* device) {
    if (device == nullptr || device[0] == '\0') {
        return "auto";
    }
    return device;
}

std::string normalize_device(const char* device) {
    const char* d = resolve_device(device);
    if (std::strcmp(d, "auto") == 0 || std::strcmp(d, "gpu") == 0) {
        return "vulkan";
    }
    return d;
}

}  // namespace

struct aicore_eloftr_options {
    std::string device = "cpu";
    int32_t threads = 0;
};

struct aicore_eloftr_ctx {
    std::unique_ptr<eloftr::EfficientLoFTRMatcher> matcher;
    std::string model_path;
    std::string device;
    std::string last_error;
};

AICORE_CAPI int aicore_eloftr_abi_version(void) { return 1; }

AICORE_CAPI aicore_eloftr_options* aicore_eloftr_options_new(void) {
    return new aicore_eloftr_options();
}

AICORE_CAPI void aicore_eloftr_options_free(aicore_eloftr_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_eloftr_options_set_device(aicore_eloftr_options* opts,
                                                  const char* device) {
    if (opts != nullptr && device != nullptr) {
        opts->device = device;
    }
}

AICORE_CAPI void aicore_eloftr_options_set_threads(aicore_eloftr_options* opts,
                                                   int n_threads) {
    if (opts != nullptr) {
        opts->threads = n_threads;
    }
}

AICORE_CAPI aicore_eloftr_ctx* aicore_eloftr_load_opts(
        const char* gguf_path, const aicore_eloftr_options* opts) {
    if (gguf_path == nullptr) {
        return nullptr;
    }
    auto* ctx = new (std::nothrow) aicore_eloftr_ctx();
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->model_path = gguf_path;
    ctx->device =
            opts != nullptr ? normalize_device(opts->device.c_str()) : "cpu";

    eloftr::EfficientLoFTROptions o;
    o.model_path = ctx->model_path;
    o.device = ctx->device;
    o.num_threads = opts != nullptr ? opts->threads : 4;

    std::string err;
    ctx->matcher = eloftr::CreateEfficientLoFTRMatcher(o, &err);
    if (!ctx->matcher) {
        ctx->last_error = err.empty() ? "failed to create ELoFTR matcher" : err;
    } else if (!ctx->matcher->Error().empty()) {
        ctx->last_error = ctx->matcher->Error();
    }
    return ctx;
}

AICORE_CAPI void aicore_eloftr_free(aicore_eloftr_ctx* ctx) { delete ctx; }

AICORE_CAPI const char* aicore_eloftr_last_error(const aicore_eloftr_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI int aicore_eloftr_match_gray(aicore_eloftr_ctx* ctx,
                                         const uint8_t* img0,
                                         const uint8_t* img1,
                                         int32_t width,
                                         int32_t height,
                                         int32_t row_stride,
                                         aicore_eloftr_match** out_matches,
                                         int32_t* out_count) {
    if (ctx == nullptr || ctx->matcher == nullptr || img0 == nullptr ||
        img1 == nullptr || out_matches == nullptr || out_count == nullptr ||
        width <= 0 || height <= 0) {
        return -1;
    }
    *out_matches = nullptr;
    *out_count = 0;

    eloftr::EfficientLoFTRResult result;
    if (!ctx->matcher->MatchGray(img0, img1, width, height, row_stride,
                                 &result)) {
        ctx->last_error = ctx->matcher->Error();
        return -1;
    }

    if (result.matches.empty()) {
        return 0;
    }

    auto* out = static_cast<aicore_eloftr_match*>(
            std::malloc(result.matches.size() * sizeof(aicore_eloftr_match)));
    if (out == nullptr) {
        ctx->last_error = "out of memory";
        return -1;
    }
    for (size_t i = 0; i < result.matches.size(); ++i) {
        out[i].x0 = result.matches[i].x0;
        out[i].y0 = result.matches[i].y0;
        out[i].x1 = result.matches[i].x1;
        out[i].y1 = result.matches[i].y1;
        out[i].score = result.matches[i].score;
    }
    *out_matches = out;
    *out_count = static_cast<int32_t>(result.matches.size());
    return 0;
}

AICORE_CAPI void aicore_eloftr_free_matches(aicore_eloftr_match* matches) {
    std::free(matches);
}

AICORE_CAPI char* aicore_eloftr_info_json(aicore_eloftr_ctx* ctx) {
    if (ctx == nullptr || ctx->matcher == nullptr) {
        return dup_cstr("{}");
    }
    const std::string json =
            std::string(
                    "{\n  \"architecture\": \"eloftr\",\n  \"device\": \"") +
            ctx->matcher->Device() + "\",\n  \"model\": \"" + ctx->model_path +
            "\"\n}";
    return dup_cstr(json);
}

AICORE_CAPI void aicore_eloftr_free_string(char* s) { std::free(s); }

AICORE_CAPI int aicore_eloftr_warmup_backend(const char* device) {
    (void)device;
    return 0;
}

AICORE_CAPI char* aicore_eloftr_model_cache_dir(void) {
    return dup_cstr(aicore::eloftr_model_cache_dir());
}

AICORE_CAPI int aicore_eloftr_quantize(const char* input_gguf,
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
