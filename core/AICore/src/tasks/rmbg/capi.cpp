// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/rmbg_capi.h"
#include "ggml_backend_utils.hpp"
#include "model_cache.hpp"
#include "rmbg.hpp"
#include "rmbg_graph.hpp"
#include "rmbg_preprocess.hpp"

namespace {

char* dup_cstr(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) {
        std::memcpy(out, s.c_str(), s.size() + 1);
    }
    return out;
}

bool read_file_bytes(const char* path,
                     std::vector<uint8_t>& out,
                     std::string* err) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        if (err) *err = std::string("cannot open file: ") + path;
        return false;
    }
    out.assign((std::istreambuf_iterator<char>(f)),
               std::istreambuf_iterator<char>());
    if (out.empty()) {
        if (err) *err = std::string("empty file: ") + path;
        return false;
    }
    return true;
}

}  // namespace

struct aicore_rmbg_options {
    std::string device = "auto";
    int32_t threads = 0;
};

struct aicore_rmbg_ctx {
    rmbg::Model model;
    std::string model_path;
    std::string device;
    int32_t threads = 0;
    std::string last_error;
    aicore_rmbg_timings timings{};
    bool has_timings = false;
};

AICORE_CAPI int aicore_rmbg_abi_version(void) { return 2; }

AICORE_CAPI aicore_rmbg_options* aicore_rmbg_options_new(void) {
    return new (std::nothrow) aicore_rmbg_options();
}

AICORE_CAPI void aicore_rmbg_options_free(aicore_rmbg_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_rmbg_options_set_device(aicore_rmbg_options* opts,
                                                const char* device) {
    if (opts != nullptr && device != nullptr) opts->device = device;
}

AICORE_CAPI void aicore_rmbg_options_set_threads(aicore_rmbg_options* opts,
                                                 int n_threads) {
    if (opts != nullptr) opts->threads = n_threads;
}

AICORE_CAPI aicore_rmbg_ctx* aicore_rmbg_load_opts(
        const char* gguf_path, const aicore_rmbg_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_rmbg_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device = opts != nullptr ? opts->device : "auto";
    ctx->threads = opts != nullptr ? opts->threads : 0;

    std::string err;
    if (!rmbg::load_gguf(gguf_path, ctx->device.c_str(), ctx->threads,
                         ctx->model, err)) {
        ctx->last_error = "failed to load RMBG-2.0 GGUF: " + err;
    }
    return ctx;
}

AICORE_CAPI void aicore_rmbg_free(aicore_rmbg_ctx* ctx) {
    if (ctx == nullptr) return;
    rmbg::free_model(ctx->model);
    delete ctx;
}

AICORE_CAPI int aicore_rmbg_is_ready(const aicore_rmbg_ctx* ctx) {
    return ctx != nullptr && ctx->model.graph_ready ? 1 : 0;
}

AICORE_CAPI const char* aicore_rmbg_last_error(const aicore_rmbg_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI int aicore_rmbg_last_timings(const aicore_rmbg_ctx* ctx,
                                         aicore_rmbg_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_timings) {
        return -1;
    }
    *out_timings = ctx->timings;
    return 0;
}

AICORE_CAPI void aicore_rmbg_free_string(char* s) { std::free(s); }

AICORE_CAPI void aicore_rmbg_free_buffer(void* p) { std::free(p); }

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsed_ms(SteadyClock::time_point start) {
    return std::chrono::duration<double, std::milli>(SteadyClock::now() - start)
            .count();
}

void begin_request(aicore_rmbg_ctx* ctx) {
    ctx->timings = {};
    ctx->has_timings = false;
}

void finish_request(aicore_rmbg_ctx* ctx, SteadyClock::time_point start) {
    ctx->timings.total_ms = elapsed_ms(start);
    ctx->has_timings = true;
}

// Shared inference core. `rgb` may be null when `encoded_bytes` is provided
// (file path / encoded image input); exactly one input must be set.
bool run_inference(aicore_rmbg_ctx* ctx,
                   const uint8_t* rgb,
                   int32_t rgb_w,
                   int32_t rgb_h,
                   const uint8_t* encoded_bytes,
                   int encoded_len,
                   std::vector<uint8_t>& rgba,
                   int& width,
                   int& height,
                   std::vector<float>& alpha,
                   std::string& err) {
    if (ctx == nullptr || !ctx->model.graph_ready) {
        err = "model not loaded";
        return false;
    }
    std::vector<float> input;
    std::vector<uint8_t> original_rgba;
    const auto preprocess_start = SteadyClock::now();
    if (rgb != nullptr) {
        if (!rmbg::decode_preprocess_rgb(
                    rgb, rgb_w, rgb_h, ctx->model.cfg.input_size,
                    ctx->model.cfg.mean, ctx->model.cfg.std, original_rgba,
                    width, height, input, err)) {
            return false;
        }
    } else {
        if (!rmbg::decode_preprocess(
                    encoded_bytes, encoded_len, ctx->model.cfg.input_size,
                    ctx->model.cfg.mean, ctx->model.cfg.std, original_rgba,
                    width, height, input, err)) {
            return false;
        }
    }
    ctx->timings.preprocess_ms = elapsed_ms(preprocess_start);
    const auto inference_start = SteadyClock::now();
    if (!ctx->model.graph->forward(input, alpha, err)) {
        return false;
    }
    ctx->timings.inference_ms = elapsed_ms(inference_start);
    rgba = std::move(original_rgba);
    return true;
}

}  // namespace

AICORE_CAPI int aicore_rmbg_remove_background_path(aicore_rmbg_ctx* ctx,
                                                   const char* image_path,
                                                   uint8_t** out_png,
                                                   int* out_len) {
    if (ctx == nullptr || image_path == nullptr || out_png == nullptr ||
        out_len == nullptr) {
        return -1;
    }
    *out_png = nullptr;
    *out_len = 0;
    begin_request(ctx);
    const auto request_start = SteadyClock::now();

    std::vector<uint8_t> bytes;
    if (!read_file_bytes(image_path, bytes, &ctx->last_error)) return -1;

    std::vector<uint8_t> rgba;
    std::vector<float> alpha;
    int width = 0, height = 0;
    std::string err;
    if (!run_inference(ctx, nullptr, 0, 0, bytes.data(),
                       static_cast<int>(bytes.size()), rgba, width, height,
                       alpha, err)) {
        ctx->last_error = err;
        return -1;
    }
    const auto postprocess_start = SteadyClock::now();
    std::vector<uint8_t> png;
    if (!rmbg::encode_result_png(rgba, width, height, alpha,
                                 ctx->model.cfg.input_size,
                                 ctx->model.cfg.input_size, png, err)) {
        ctx->last_error = err;
        return -1;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(png.size()));
    if (buf == nullptr) {
        ctx->last_error = "output allocation failed";
        return -1;
    }
    std::memcpy(buf, png.data(), png.size());
    *out_png = buf;
    *out_len = static_cast<int>(png.size());
    ctx->timings.postprocess_ms = elapsed_ms(postprocess_start);
    finish_request(ctx, request_start);
    return 0;
}

AICORE_CAPI int aicore_rmbg_remove_background_rgb(aicore_rmbg_ctx* ctx,
                                                  const uint8_t* rgb,
                                                  int32_t width,
                                                  int32_t height,
                                                  uint8_t** out_png,
                                                  int* out_len) {
    if (ctx == nullptr || rgb == nullptr || width <= 0 || height <= 0 ||
        out_png == nullptr || out_len == nullptr) {
        return -1;
    }
    *out_png = nullptr;
    *out_len = 0;
    begin_request(ctx);
    const auto request_start = SteadyClock::now();

    std::vector<uint8_t> rgba;
    std::vector<float> alpha;
    int out_w = 0, out_h = 0;
    std::string err;
    if (!run_inference(ctx, rgb, width, height, nullptr, 0, rgba, out_w, out_h,
                       alpha, err)) {
        ctx->last_error = err;
        return -1;
    }
    const auto postprocess_start = SteadyClock::now();
    std::vector<uint8_t> png;
    if (!rmbg::encode_result_png(rgba, out_w, out_h, alpha,
                                 ctx->model.cfg.input_size,
                                 ctx->model.cfg.input_size, png, err)) {
        ctx->last_error = err;
        return -1;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(png.size()));
    if (buf == nullptr) {
        ctx->last_error = "output allocation failed";
        return -1;
    }
    std::memcpy(buf, png.data(), png.size());
    *out_png = buf;
    *out_len = static_cast<int>(png.size());
    ctx->timings.postprocess_ms = elapsed_ms(postprocess_start);
    finish_request(ctx, request_start);
    return 0;
}

AICORE_CAPI int aicore_rmbg_remove_background_rgba(aicore_rmbg_ctx* ctx,
                                                   const uint8_t* rgb,
                                                   int32_t width,
                                                   int32_t height,
                                                   uint8_t** out_rgba,
                                                   int32_t* out_width,
                                                   int32_t* out_height,
                                                   int* out_len) {
    if (ctx == nullptr || rgb == nullptr || width <= 0 || height <= 0 ||
        out_rgba == nullptr || out_width == nullptr || out_height == nullptr ||
        out_len == nullptr) {
        return -1;
    }
    *out_rgba = nullptr;
    *out_width = 0;
    *out_height = 0;
    *out_len = 0;
    begin_request(ctx);
    const auto request_start = SteadyClock::now();

    std::vector<uint8_t> rgba;
    std::vector<float> alpha;
    int out_w = 0, out_h = 0;
    std::string err;
    if (!run_inference(ctx, rgb, width, height, nullptr, 0, rgba, out_w, out_h,
                       alpha, err)) {
        ctx->last_error = err;
        return -1;
    }
    const auto postprocess_start = SteadyClock::now();
    /* Same composite as the PNG path (bicubic alpha upsampled in-place), but
     * without the encode/decode round-trip for in-memory consumers. */
    if (!rmbg::compose_alpha(rgba, out_w, out_h, alpha,
                             ctx->model.cfg.input_size,
                             ctx->model.cfg.input_size, err)) {
        ctx->last_error = err;
        return -1;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(rgba.size()));
    if (buf == nullptr) {
        ctx->last_error = "output allocation failed";
        return -1;
    }
    std::memcpy(buf, rgba.data(), rgba.size());
    *out_rgba = buf;
    *out_width = out_w;
    *out_height = out_h;
    *out_len = static_cast<int>(rgba.size());
    ctx->timings.postprocess_ms = elapsed_ms(postprocess_start);
    finish_request(ctx, request_start);
    return 0;
}

AICORE_CAPI int aicore_rmbg_alpha_mat_rgb(aicore_rmbg_ctx* ctx,
                                          const uint8_t* rgb,
                                          int32_t width,
                                          int32_t height,
                                          uint8_t** out_alpha,
                                          int32_t* out_width,
                                          int32_t* out_height) {
    if (ctx == nullptr || rgb == nullptr || width <= 0 || height <= 0 ||
        out_alpha == nullptr || out_width == nullptr || out_height == nullptr) {
        return -1;
    }
    *out_alpha = nullptr;
    *out_width = 0;
    *out_height = 0;
    begin_request(ctx);
    const auto request_start = SteadyClock::now();

    std::vector<uint8_t> rgba;
    std::vector<float> alpha;
    int out_w = 0, out_h = 0;
    std::string err;
    if (!run_inference(ctx, rgb, width, height, nullptr, 0, rgba, out_w, out_h,
                       alpha, err)) {
        ctx->last_error = err;
        return -1;
    }
    const auto postprocess_start = SteadyClock::now();
    std::vector<uint8_t> alpha8;
    if (!rmbg::upsample_alpha(alpha, ctx->model.cfg.input_size,
                              ctx->model.cfg.input_size, out_w, out_h, alpha8,
                              err)) {
        ctx->last_error = err;
        return -1;
    }
    uint8_t* buf = static_cast<uint8_t*>(std::malloc(alpha8.size()));
    if (buf == nullptr) {
        ctx->last_error = "output allocation failed";
        return -1;
    }
    std::memcpy(buf, alpha8.data(), alpha8.size());
    *out_alpha = buf;
    *out_width = out_w;
    *out_height = out_h;
    ctx->timings.postprocess_ms = elapsed_ms(postprocess_start);
    finish_request(ctx, request_start);
    return 0;
}

AICORE_CAPI char* aicore_rmbg_info_json(aicore_rmbg_ctx* ctx) {
    if (ctx == nullptr || !ctx->model.graph_ready) return nullptr;
    std::ostringstream o;
    o << "{\"model\":\"RMBG-2.0 (BiRefNet-Swin-L)\","
      << "\"input_size\":" << ctx->model.cfg.input_size << ","
      << "\"backend\":\"" << ctx->model.backend_name << "\","
      << "\"math_profile\":\"" << ctx->model.math_profile << "\","
      << "\"device\":\"" << ctx->device << "\","
      << "\"threads\":" << ctx->threads << "}";
    return dup_cstr(o.str());
}

AICORE_CAPI int aicore_rmbg_warmup_backend(const char* device) {
    rmbg::configure_backend_profile(device != nullptr ? device : "auto");
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_rmbg_shutdown(void) {}

AICORE_CAPI char* aicore_rmbg_model_cache_dir(void) {
    return dup_cstr(aicore::rmbg_model_cache_dir());
}
