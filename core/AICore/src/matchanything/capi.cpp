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

#include "aicore/matchanything_capi.h"
#include "gguf_weight_quantize.hpp"
#include "matchanything.hpp"
#include "model_cache.hpp"

namespace {

char *dup_cstr(const std::string &s) {
    char *out = static_cast<char *>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

std::string normalize_device(const char *device) {
    if (device == nullptr || device[0] == '\0' ||
        std::strcmp(device, "auto") == 0 || std::strcmp(device, "gpu") == 0) {
        return "vulkan";
    }
    return device;
}

matchanything::Variant to_variant(aicore_matchanything_variant v) {
    return v == AICORE_MATCHANYTHING_ROMA ? matchanything::Variant::kRoma
                                          : matchanything::Variant::kEloftr;
}

}  // namespace

struct aicore_matchanything_options {
    std::string device = "cpu";
    int32_t threads = 0;
    aicore_matchanything_variant variant = AICORE_MATCHANYTHING_ELOFTR;
};

struct aicore_matchanything_ctx {
    std::unique_ptr<matchanything::MatchAnythingMatcher> matcher;
    std::string model_path;
    std::string device;
    aicore_matchanything_variant variant = AICORE_MATCHANYTHING_ELOFTR;
    std::string last_error;
};

AICORE_CAPI int aicore_matchanything_abi_version(void) { return 1; }

AICORE_CAPI aicore_matchanything_options *aicore_matchanything_options_new(
        void) {
    return new aicore_matchanything_options();
}

AICORE_CAPI void aicore_matchanything_options_free(
        aicore_matchanything_options *opts) {
    delete opts;
}

AICORE_CAPI void aicore_matchanything_options_set_device(
        aicore_matchanything_options *opts, const char *device) {
    if (opts != nullptr && device != nullptr) {
        opts->device = device;
    }
}

AICORE_CAPI void aicore_matchanything_options_set_threads(
        aicore_matchanything_options *opts, int n_threads) {
    if (opts != nullptr) {
        opts->threads = n_threads;
    }
}

AICORE_CAPI void aicore_matchanything_options_set_variant(
        aicore_matchanything_options *opts,
        aicore_matchanything_variant variant) {
    if (opts != nullptr) {
        opts->variant = variant;
    }
}

AICORE_CAPI aicore_matchanything_ctx *aicore_matchanything_load_opts(
        const char *gguf_path, const aicore_matchanything_options *opts) {
    if (gguf_path == nullptr) {
        return nullptr;
    }
    auto *ctx = new (std::nothrow) aicore_matchanything_ctx();
    if (ctx == nullptr) {
        return nullptr;
    }
    ctx->model_path = gguf_path;
    ctx->device =
            opts != nullptr ? normalize_device(opts->device.c_str()) : "cpu";
    ctx->variant =
            opts != nullptr ? opts->variant : AICORE_MATCHANYTHING_ELOFTR;

    matchanything::MatchAnythingOptions o;
    o.model_path = ctx->model_path;
    o.device = ctx->device;
    o.num_threads = opts != nullptr ? opts->threads : 4;
    o.variant = to_variant(ctx->variant);

    std::string err;
    ctx->matcher = matchanything::CreateMatchAnythingMatcher(o, &err);
    if (!ctx->matcher) {
        ctx->last_error =
                err.empty() ? "failed to create MatchAnything matcher" : err;
    } else if (!ctx->matcher->Error().empty()) {
        ctx->last_error = ctx->matcher->Error();
    }
    return ctx;
}

AICORE_CAPI void aicore_matchanything_free(aicore_matchanything_ctx *ctx) {
    delete ctx;
}

AICORE_CAPI const char *aicore_matchanything_last_error(
        const aicore_matchanything_ctx *ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI int aicore_matchanything_match_gray(
        aicore_matchanything_ctx *ctx,
        const uint8_t *img0,
        const uint8_t *img1,
        int32_t width,
        int32_t height,
        int32_t row_stride,
        aicore_matchanything_match **out_matches,
        int32_t *out_count) {
    if (ctx == nullptr || ctx->matcher == nullptr || img0 == nullptr ||
        img1 == nullptr || out_matches == nullptr || out_count == nullptr ||
        width <= 0 || height <= 0) {
        return -1;
    }
    *out_matches = nullptr;
    *out_count = 0;

    matchanything::MatchAnythingResult result;
    if (!ctx->matcher->MatchGray(img0, img1, width, height, row_stride,
                                 &result)) {
        ctx->last_error = ctx->matcher->Error();
        return -1;
    }

    if (result.matches.empty()) {
        return 0;
    }

    auto *out = static_cast<aicore_matchanything_match *>(std::malloc(
            result.matches.size() * sizeof(aicore_matchanything_match)));
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

AICORE_CAPI void aicore_matchanything_free_matches(
        aicore_matchanything_match *matches) {
    std::free(matches);
}

AICORE_CAPI char *aicore_matchanything_info_json(
        aicore_matchanything_ctx *ctx) {
    if (ctx == nullptr || ctx->matcher == nullptr) {
        return dup_cstr("{}");
    }
    const std::string json = std::string(
                                     "{\n  \"architecture\": "
                                     "\"matchanything\",\n  \"variant\": \"") +
                             ctx->matcher->VariantName() +
                             "\",\n  \"device\": \"" + ctx->matcher->Device() +
                             "\",\n  \"model\": \"" + ctx->model_path + "\"\n}";
    return dup_cstr(json);
}

AICORE_CAPI void aicore_matchanything_free_string(char *s) { std::free(s); }

AICORE_CAPI int aicore_matchanything_warmup_backend(const char *device) {
    (void)device;
    return 0;
}

AICORE_CAPI char *aicore_matchanything_model_cache_dir(void) {
    return dup_cstr(aicore::matchanything_model_cache_dir());
}

AICORE_CAPI int aicore_matchanything_quantize(const char *input_gguf,
                                              const char *output_gguf,
                                              const char *type) {
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
