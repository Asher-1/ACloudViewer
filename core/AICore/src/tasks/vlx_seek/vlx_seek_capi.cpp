// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/vlx_seek_capi.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "aicore/runtime_capi.h"
#include "common/model_cache.hpp"
#include "tasks/vlx_seek/third_party/stb/stb_image.h"
#include "tasks/vlx_seek/vlx_seek_runtime.h"

using vlx::VlxRunRequest;
using vlx::VlxRunResult;
using vlx::VlxSeekOptions;
using vlx::VlxSeekRuntime;

struct aicore_vlx_seek_options {
    VlxSeekOptions opts;
    std::string device;
    std::string precision = "f16";
    std::string models_dir;
    std::string wedetect_gguf, vision_gguf, llm_gguf;
    int log_level = 1;
};

struct aicore_vlx_seek_ctx {
    std::unique_ptr<VlxSeekRuntime> runtime;
    VlxRunResult last;  // retained typed result of the most recent run
    bool has_last = false;
};

int aicore_vlx_seek_abi_version(void) { return AICORE_VLX_SEEK_ABI_VERSION; }

aicore_vlx_seek_options* aicore_vlx_seek_options_new(void) {
    return new (std::nothrow) aicore_vlx_seek_options();
}

void aicore_vlx_seek_options_free(aicore_vlx_seek_options* opts) {
    delete opts;
}

void aicore_vlx_seek_options_set_device(aicore_vlx_seek_options* opts,
                                        const char* device) {
    if (opts && device) opts->device = device;
}

void aicore_vlx_seek_options_set_threads(aicore_vlx_seek_options* opts,
                                         int n_threads) {
    if (opts) opts->opts.threads = n_threads;
}

void aicore_vlx_seek_options_set_precision(aicore_vlx_seek_options* opts,
                                           const char* precision) {
    if (opts && precision) opts->precision = precision;
}

void aicore_vlx_seek_options_set_models_dir(aicore_vlx_seek_options* opts,
                                            const char* dir) {
    if (opts && dir) opts->models_dir = dir;
}

void aicore_vlx_seek_options_set_wedetect_gguf(aicore_vlx_seek_options* opts,
                                               const char* path) {
    if (opts) opts->wedetect_gguf = path ? path : "";
}

void aicore_vlx_seek_options_set_vision_gguf(aicore_vlx_seek_options* opts,
                                             const char* path) {
    if (opts) opts->vision_gguf = path ? path : "";
}

void aicore_vlx_seek_options_set_llm_gguf(aicore_vlx_seek_options* opts,
                                          const char* path) {
    if (opts) opts->llm_gguf = path ? path : "";
}

void aicore_vlx_seek_options_set_max_new_tokens(aicore_vlx_seek_options* opts,
                                                int n_tokens) {
    if (opts) opts->opts.max_new_tokens = n_tokens;
}

void aicore_vlx_seek_options_set_log_level(aicore_vlx_seek_options* opts,
                                           int log_level) {
    if (opts) opts->log_level = log_level;
}

void aicore_vlx_seek_options_set_dump_dir(aicore_vlx_seek_options* opts,
                                          const char* dir) {
    if (opts) opts->opts.dump_dir = dir ? dir : "";
}

aicore_vlx_seek_ctx* aicore_vlx_seek_load_opts(
        const aicore_vlx_seek_options* opts) {
    if (!opts) return nullptr;
    aicore_vlx_seek_ctx* ctx = new (std::nothrow) aicore_vlx_seek_ctx();
    if (!ctx) return nullptr;
    VlxSeekOptions o = opts->opts;
    o.device = opts->device;
    o.precision = opts->precision;
    o.models_dir = opts->models_dir;
    o.wedetect_gguf = opts->wedetect_gguf;
    o.vision_gguf = opts->vision_gguf;
    o.llm_gguf = opts->llm_gguf;
    ctx->runtime = std::make_unique<VlxSeekRuntime>(o);
    if (!ctx->runtime->load()) {
        ctx->last.error = ctx->runtime->last_error();
        return ctx;  // is_ready()==0; caller inspects last_error
    }
    return ctx;
}

void aicore_vlx_seek_free(aicore_vlx_seek_ctx* ctx) { delete ctx; }

int aicore_vlx_seek_is_ready(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime && ctx->runtime->ready() ? 1 : 0;
}

const char* aicore_vlx_seek_last_error(const aicore_vlx_seek_ctx* ctx) {
    if (!ctx) return nullptr;
    if (!ctx->last.error.empty()) return ctx->last.error.c_str();
    return ctx->runtime && !ctx->runtime->last_error().empty()
                   ? ctx->runtime->last_error().c_str()
                   : nullptr;
}

void aicore_vlx_seek_free_buffer(void* p) { free(p); }

int aicore_vlx_seek_load_path_rgb(const char* image_path,
                                  uint8_t** out_rgb,
                                  int32_t* out_width,
                                  int32_t* out_height) {
    if (!image_path || !out_rgb || !out_width || !out_height) return -1;
    *out_rgb = nullptr;
    // stb_image decodes to tightly-packed RGB8 when requested (n=3); the
    // implementation lives in vlx_image_prep.cpp — a private detail, only the
    // decoded buffer crosses here.
    int w = 0, h = 0, n = 0;
    uint8_t* p = stbi_load(image_path, &w, &h, &n, 3);
    if (!p) return -2;
    *out_rgb = p;
    *out_width = w;
    *out_height = h;
    return 0;
}

// Converts any supported view layout into tightly-packed RGB8.
static bool to_rgb8(const aicore_image_view* v, std::vector<uint8_t>* out) {
    if (!v || !v->data || v->width <= 0 || v->height <= 0) return false;
    const int w = v->width, h = v->height;
    out->assign(size_t(w) * h * 3, 0);
    auto px = [&](int x, int y, uint8_t* r, uint8_t* g, uint8_t* b) {
        const uint8_t* row = v->data + size_t(y) * v->row_stride_bytes;
        switch (v->format) {
            case AICORE_IMAGE_RGB8:
                *r = row[size_t(x) * 3];
                *g = row[size_t(x) * 3 + 1];
                *b = row[size_t(x) * 3 + 2];
                break;
            case AICORE_IMAGE_RGBA8:
                *r = row[size_t(x) * 4];
                *g = row[size_t(x) * 4 + 1];
                *b = row[size_t(x) * 4 + 2];
                break;
            case AICORE_IMAGE_BGR8:
                *b = row[size_t(x) * 3];
                *g = row[size_t(x) * 3 + 1];
                *r = row[size_t(x) * 3 + 2];
                break;
            case AICORE_IMAGE_BGRA8:
                *b = row[size_t(x) * 4];
                *g = row[size_t(x) * 4 + 1];
                *r = row[size_t(x) * 4 + 2];
                break;
            case AICORE_IMAGE_GRAY8:
                *r = *g = *b = row[x];
                break;
            default:
                return false;
        }
        return true;
    };
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            uint8_t r, g, b;
            if (!px(x, y, &r, &g, &b)) return false;
            uint8_t* dst = out->data() + (size_t(y) * w + x) * 3;
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        }
    return true;
}

int aicore_vlx_seek_run_image(aicore_vlx_seek_ctx* ctx,
                              const aicore_image_view* img,
                              const char* task,
                              const char* text,
                              const char* lang,
                              const float* boxes,
                              int32_t n_boxes,
                              const int32_t* target_indexes,
                              int32_t n_targets) {
    if (!ctx || !ctx->runtime || !img || !task || !text || !lang) return -1;
    std::vector<uint8_t> rgb;
    if (!to_rgb8(img, &rgb)) return -1;

    VlxRunRequest req;
    req.rgb = rgb.data();
    req.width = img->width;
    req.height = img->height;
    req.task = task;
    req.text = text;
    req.lang = lang ? lang : "en";
    req.boxes = boxes;
    req.n_boxes = boxes ? n_boxes : 0;
    req.target_indexes = target_indexes;
    req.n_targets = target_indexes ? n_targets : 0;
    VlxRunResult r = ctx->runtime->run(req);
    ctx->last = std::move(r);
    ctx->has_last = true;
    return ctx->last.code;
}

const char* aicore_vlx_seek_result_answer(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.answer.c_str() : nullptr;
}

char* aicore_vlx_seek_result_json(const aicore_vlx_seek_ctx* ctx) {
    if (!ctx || !ctx->has_last) return nullptr;
    const std::string& js = ctx->last.json;
    char* out = static_cast<char*>(malloc(js.size() + 1));
    if (out) std::memcpy(out, js.c_str(), js.size() + 1);
    return out;
}

int32_t aicore_vlx_seek_result_box_count(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? int32_t(ctx->last.boxes.size()) : 0;
}

int aicore_vlx_seek_result_box(const aicore_vlx_seek_ctx* ctx,
                               int32_t index,
                               float* out) {
    if (!ctx || !ctx->has_last || index < 0 ||
        size_t(index) >= ctx->last.boxes.size() || !out)
        return -1;
    const auto& b = ctx->last.boxes[size_t(index)].box;
    out[0] = float(b[0]);
    out[1] = float(b[1]);
    out[2] = float(b[2]);
    out[3] = float(b[3]);
    return 0;
}

const char* aicore_vlx_seek_result_box_label(const aicore_vlx_seek_ctx* ctx,
                                             int32_t index) {
    if (!ctx || !ctx->has_last || index < 0 ||
        size_t(index) >= ctx->last.boxes.size())
        return nullptr;
    return ctx->last.boxes[size_t(index)].label.c_str();
}

int32_t aicore_vlx_seek_result_prompt_tokens(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.prompt_tokens : 0;
}

int32_t aicore_vlx_seek_result_completion_tokens(
        const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.completion_tokens : 0;
}

int32_t aicore_vlx_seek_result_proposal_count(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->has_last ? ctx->last.proposal_count : 0;
}

int aicore_vlx_seek_result_render_png(const aicore_vlx_seek_ctx* ctx,
                                      const aicore_image_view* img,
                                      const char* out_png_path) {
    if (!ctx || !ctx->has_last || !img || !out_png_path) return -1;
    std::vector<uint8_t> rgb;
    if (!to_rgb8(img, &rgb)) return -1;
    return ctx->runtime->render_png(ctx->last, rgb.data(), img->width,
                                    img->height, out_png_path)
                   ? 0
                   : -2;
}

int aicore_vlx_seek_last_pipeline_timings(const aicore_vlx_seek_ctx* ctx,
                                          aicore_pipeline_timings* out) {
    if (!ctx || !ctx->has_last || !out) return -1;
    out->abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    out->valid_fields = AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
                        AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    if (ctx->last.serialization_ms > 0)
        out->valid_fields |= AICORE_TIMING_SERIALIZATION;
    out->preprocess_ms = ctx->last.preprocess_ms;
    out->inference_ms = ctx->last.inference_ms;
    out->postprocess_ms = ctx->last.postprocess_ms;
    out->serialization_ms = ctx->last.serialization_ms;
    out->e2e_ms = ctx->last.e2e_ms;
    return 0;
}

const char* aicore_vlx_seek_context_device(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime ? ctx->runtime->device().c_str() : nullptr;
}

int aicore_vlx_seek_context_threads(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime ? ctx->runtime->threads() : 0;
}

const char* aicore_vlx_seek_context_precision(const aicore_vlx_seek_ctx* ctx) {
    return ctx && ctx->runtime ? ctx->runtime->precision().c_str() : nullptr;
}

char* aicore_vlx_seek_info_json(aicore_vlx_seek_ctx* ctx) {
    if (!ctx || !ctx->runtime) return nullptr;
    const std::string js = ctx->runtime->info_json();
    char* out = static_cast<char*>(malloc(js.size() + 1));
    if (out) std::memcpy(out, js.c_str(), js.size() + 1);
    return out;
}

void aicore_vlx_seek_shutdown(void) {
    // Delegates to aicore_runtime_shutdown(); never destroys live contexts.
    aicore_runtime_shutdown();
}

char* aicore_vlx_seek_model_cache_dir(void) {
    const std::string dir = aicore::vlx_seek_model_cache_dir();
    char* out = static_cast<char*>(malloc(dir.size() + 1));
    if (out) std::memcpy(out, dir.c_str(), dir.size() + 1);
    return out;
}
