// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include "aicore/wevisone_capi.h"

#include <cstring>
#include <memory>
#include <new>
#include <sstream>
#include <string>

#include "aicore/runtime_capi.h"
#include "common/model_cache.hpp"
#include "tasks/wevisone/wevisone_image_prep.h"
#include "tasks/wevisone/wevisone_pipeline_common.h"
#include "tasks/wevisone/wevisone_render.h"
#include "tasks/wevisone/wevisone_runtime.h"

#define WWV_LOG_INFO(...) ::wwv::logf(wwv::LogLevel::kInfo, __VA_ARGS__)

namespace wwv {

namespace {

int map_rt_error(const std::string& err) {
    if (err.find("not compiled") != std::string::npos ||
        err.find("cannot hold") != std::string::npos ||
        err.find("insufficient") != std::string::npos ||
        err.find("unavailable") != std::string::npos ||
        err.find("no ggml device") != std::string::npos) {
        return -3;  // resources
    }
    if (err.find("invalid") != std::string::npos ||
        err.find("unsupported") != std::string::npos) {
        return -1;  // contract
    }
    return -2;  // runtime
}

// Shared by aicore_wevisone_result_json and the per-index json getter.
char* render_result_json(const RunResult& res) {
    std::ostringstream json;
    json << "{\"answer\": \"" << json_escape(res.answer) << "\", ";
    json << "\"boxes\": [";
    for (size_t i = 0; i < res.boxes.size(); ++i) {
        const ResultBox& b = res.boxes[i];
        json << (i > 0 ? ", " : "") << "[" << b.x1 << ", " << b.y1 << ", "
             << b.x2 << ", " << b.y2 << "]";
    }
    json << "], ";
    json << "\"has_masks\": " << (res.masks_decoded ? "true" : "false") << ", ";
    json << "\"prompt_tokens\": " << res.prompt_tokens << ", ";
    json << "\"completion_tokens\": " << res.completion_tokens << "}";
    const std::string s = json.str();
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

}  // namespace

}  // namespace wwv

// The C API declares aicore_wevisone_options / aicore_wevisone_ctx as opaque
// typedefs; the concrete layouts live here at global scope and never cross the
// ABI.
struct aicore_wevisone_options {
    wwv::RuntimeOptions rt;
};

struct aicore_wevisone_ctx {
    std::unique_ptr<wwv::WevisoneRuntime> rt;
    // Load-time failure message (rt is null when create() failed); keeps
    // last_error readable on a not-ready context (vlx_seek pattern).
    std::string load_error;
    wwv::RunResult last;
    // Results of the most recent run/run_batch (engine.predict_batch
    // contract): one entry per image, cleared on every run call. The
    // single-image getters above keep reading `last` so existing ABI users
    // are untouched; the indexed getters read this vector.
    std::vector<wwv::RunResult> batch;
};

using namespace wwv;  // NOLINT(google-build-using-namespace) — C shim only

extern "C" {

int aicore_wevisone_abi_version(void) { return AICORE_WEVISONE_ABI_VERSION; }

aicore_wevisone_options* aicore_wevisone_options_new(void) {
    return new (std::nothrow) aicore_wevisone_options();
}

void aicore_wevisone_options_free(aicore_wevisone_options* opts) {
    delete opts;
}

void aicore_wevisone_options_set_device(aicore_wevisone_options* opts,
                                        const char* device) {
    if (opts != nullptr && device != nullptr) opts->rt.device = device;
}

void aicore_wevisone_options_set_threads(aicore_wevisone_options* opts,
                                         int n_threads) {
    if (opts != nullptr) opts->rt.threads = n_threads;
}

void aicore_wevisone_options_set_size(aicore_wevisone_options* opts,
                                      const char* size) {
    if (opts != nullptr && size != nullptr) opts->rt.size = size;
}

void aicore_wevisone_options_set_precision(aicore_wevisone_options* opts,
                                           const char* precision) {
    if (opts != nullptr && precision != nullptr) opts->rt.precision = precision;
}

void aicore_wevisone_options_set_decoder_precision(
        aicore_wevisone_options* opts, const char* precision) {
    if (opts != nullptr && precision != nullptr) {
        opts->rt.decoder_precision = precision;
    }
}

void aicore_wevisone_options_set_models_dir(aicore_wevisone_options* opts,
                                            const char* dir) {
    if (opts != nullptr && dir != nullptr) opts->rt.models_dir = dir;
}

void aicore_wevisone_options_set_vlm_gguf(aicore_wevisone_options* opts,
                                          const char* path) {
    if (opts != nullptr && path != nullptr) opts->rt.vlm_gguf = path;
}

void aicore_wevisone_options_set_mmproj_gguf(aicore_wevisone_options* opts,
                                             const char* path) {
    if (opts != nullptr && path != nullptr) opts->rt.mmproj_gguf = path;
}

void aicore_wevisone_options_set_mkl_log_lib(aicore_wevisone_options* opts,
                                             const char* path) {
    if (opts != nullptr && path != nullptr) opts->rt.mkl_log_lib = path;
}

void aicore_wevisone_options_set_decoder_gguf(aicore_wevisone_options* opts,
                                              const char* path) {
    if (opts != nullptr && path != nullptr) opts->rt.decoder_gguf = path;
}

void aicore_wevisone_options_set_max_new_tokens(aicore_wevisone_options* opts,
                                                int n_tokens) {
    if (opts != nullptr) opts->rt.max_new_tokens = n_tokens;
}

void aicore_wevisone_options_set_decode_masks(aicore_wevisone_options* opts,
                                              int enable) {
    if (opts != nullptr) opts->rt.decode_masks = enable != 0;
}

void aicore_wevisone_options_set_log_level(aicore_wevisone_options* opts,
                                           int log_level) {
    if (opts != nullptr) opts->rt.log_level = log_level;
}

void aicore_wevisone_options_set_dump_dir(aicore_wevisone_options* opts,
                                          const char* dir) {
    if (opts != nullptr && dir != nullptr) opts->rt.dump_dir = dir;
}

aicore_wevisone_ctx* aicore_wevisone_load_opts(
        const aicore_wevisone_options* opts) {
    if (opts == nullptr) return nullptr;
    auto ctx = new (std::nothrow) aicore_wevisone_ctx();
    if (ctx == nullptr) return nullptr;
    std::string err;
    ctx->rt = WevisoneRuntime::create(opts->rt, &err);
    if (ctx->rt == nullptr) {
        WWV_LOG_INFO("load failed: %s", err.c_str());
        ctx->load_error = err.empty() ? "model load failed" : err;
        return ctx;  // is_ready()==0; caller inspects last_error
    }
    return ctx;
}

void aicore_wevisone_free(aicore_wevisone_ctx* ctx) { delete ctx; }

int aicore_wevisone_is_ready(const aicore_wevisone_ctx* ctx) {
    return ctx != nullptr && ctx->rt != nullptr && ctx->rt->is_ready() ? 1 : 0;
}

const char* aicore_wevisone_last_error(const aicore_wevisone_ctx* ctx) {
    if (ctx == nullptr) return nullptr;
    if (!ctx->load_error.empty()) return ctx->load_error.c_str();
    return ctx->rt == nullptr || ctx->rt->last_error() == nullptr ||
                           ctx->rt->last_error()[0] == '\0'
                   ? nullptr
                   : ctx->rt->last_error();
}

void aicore_wevisone_free_buffer(void* p) { std::free(p); }

int aicore_wevisone_load_path_rgb(const char* image_path,
                                  uint8_t** out_rgb,
                                  int32_t* out_width,
                                  int32_t* out_height) {
    if (image_path == nullptr || out_rgb == nullptr || out_width == nullptr ||
        out_height == nullptr) {
        return -1;
    }
    RgbImage img;
    std::string err;
    if (!load_path_rgb(image_path, &img, &err)) return -1;
    const size_t n = static_cast<size_t>(img.width) * img.height * 3;
    *out_rgb = static_cast<uint8_t*>(std::malloc(n));
    if (*out_rgb == nullptr) return -1;
    std::memcpy(*out_rgb, img.pixels.data(), n);
    *out_width = img.width;
    *out_height = img.height;
    return 0;
}

int aicore_wevisone_run_image(aicore_wevisone_ctx* ctx,
                              const aicore_image_view* img,
                              const char* task,
                              const char* query,
                              const float* ref_boxes,
                              int32_t n_ref_boxes) {
    if (ctx == nullptr || ctx->rt == nullptr || img == nullptr ||
        task == nullptr) {
        return -1;
    }
    if (img->data == nullptr || img->width <= 0 || img->height <= 0) {
        return -1;
    }
    RunResult res;
    if (!ctx->rt->run_image(*img, task, query != nullptr ? query : "",
                            ref_boxes, n_ref_boxes, &res)) {
        ctx->batch.clear();
        return map_rt_error(ctx->rt->last_error());
    }
    ctx->batch.clear();
    ctx->batch.push_back(res);
    ctx->last = std::move(res);
    return 0;
}

int aicore_wevisone_run_batch(aicore_wevisone_ctx* ctx,
                              const aicore_image_view* imgs,
                              int32_t n_images,
                              const char* task,
                              const char* query,
                              const float* ref_boxes,
                              int32_t n_ref_boxes) {
    if (ctx == nullptr || ctx->rt == nullptr || imgs == nullptr ||
        task == nullptr || n_images <= 0) {
        return -1;
    }
    ctx->batch.clear();
    ctx->batch.reserve(static_cast<size_t>(n_images));
    for (int32_t i = 0; i < n_images; ++i) {
        const aicore_image_view& img = imgs[i];
        if (img.data == nullptr || img.width <= 0 || img.height <= 0) {
            return -1;
        }
        RunResult res;
        if (!ctx->rt->run_image(img, task, query != nullptr ? query : "",
                                ref_boxes, n_ref_boxes, &res)) {
            // Abort on the first failing image, matching the python
            // engine.predict_batch exception semantics (the exception
            // surfaces; already-produced results stay readable).
            ctx->last = std::move(res);
            return map_rt_error(ctx->rt->last_error());
        }
        ctx->batch.push_back(std::move(res));
    }
    // Single-image getters surface the last batch entry; the indexed
    // getters below are the batch contract.
    ctx->last = ctx->batch.back();
    return 0;
}

int aicore_wevisone_run_decode_masks(aicore_wevisone_ctx* ctx,
                                     const aicore_image_view* img,
                                     const char* prompt,
                                     const char* response,
                                     const float* boxes01,
                                     int32_t n_boxes) {
    if (ctx == nullptr || ctx->rt == nullptr || img == nullptr ||
        response == nullptr) {
        return -1;
    }
    if (img->data == nullptr || img->width <= 0 || img->height <= 0) {
        return -1;
    }
    RunResult res;
    if (!ctx->rt->run_decode_masks(*img, prompt != nullptr ? prompt : "",
                                   response, boxes01, n_boxes, &res)) {
        ctx->batch.clear();
        return map_rt_error(ctx->rt->last_error());
    }
    ctx->batch.clear();
    ctx->batch.push_back(res);
    ctx->last = std::move(res);
    return 0;
}

const char* aicore_wevisone_result_answer(const aicore_wevisone_ctx* ctx) {
    return ctx == nullptr ? nullptr : ctx->last.answer.c_str();
}

char* aicore_wevisone_result_json(const aicore_wevisone_ctx* ctx) {
    if (ctx == nullptr) return nullptr;
    return render_result_json(ctx->last);
}

int32_t aicore_wevisone_result_box_count(const aicore_wevisone_ctx* ctx) {
    return ctx == nullptr ? 0 : static_cast<int32_t>(ctx->last.boxes.size());
}

int aicore_wevisone_result_box(const aicore_wevisone_ctx* ctx,
                               int32_t index,
                               float* out) {
    if (ctx == nullptr || out == nullptr || index < 0 ||
        index >= static_cast<int32_t>(ctx->last.boxes.size())) {
        return -1;
    }
    const ResultBox& b = ctx->last.boxes[static_cast<size_t>(index)];
    out[0] = b.x1;
    out[1] = b.y1;
    out[2] = b.x2;
    out[3] = b.y2;
    return 0;
}

int aicore_wevisone_result_has_masks(const aicore_wevisone_ctx* ctx) {
    return ctx != nullptr && ctx->last.masks_decoded ? 1 : 0;
}

int32_t aicore_wevisone_result_matte_count(const aicore_wevisone_ctx* ctx) {
    return aicore_wevisone_result_has_masks(ctx) != 0
                   ? aicore_wevisone_result_box_count(ctx)
                   : 0;
}

int aicore_wevisone_result_matte_size(const aicore_wevisone_ctx* ctx,
                                      int32_t index,
                                      int32_t* out_width,
                                      int32_t* out_height) {
    if (ctx == nullptr || out_width == nullptr || out_height == nullptr ||
        index != 0 || !ctx->last.masks_decoded) {
        return -1;
    }
    *out_width = ctx->last.matte_width;
    *out_height = ctx->last.matte_height;
    return 0;
}

const float* aicore_wevisone_result_matte_data(const aicore_wevisone_ctx* ctx,
                                               int32_t index) {
    if (ctx == nullptr || index != 0 || !ctx->last.masks_decoded)
        return nullptr;
    return ctx->last.matte.data();
}

int32_t aicore_wevisone_result_prompt_tokens(const aicore_wevisone_ctx* ctx) {
    return ctx == nullptr ? 0 : ctx->last.prompt_tokens;
}

int32_t aicore_wevisone_result_completion_tokens(
        const aicore_wevisone_ctx* ctx) {
    return ctx == nullptr ? 0 : ctx->last.completion_tokens;
}

int32_t aicore_wevisone_result_count(const aicore_wevisone_ctx* ctx) {
    if (ctx == nullptr || ctx->batch.empty()) return 0;
    return static_cast<int32_t>(ctx->batch.size());
}

static const RunResult* batch_result(const aicore_wevisone_ctx* ctx,
                                     int32_t index) {
    if (ctx == nullptr || index < 0 ||
        index >= static_cast<int32_t>(ctx->batch.size())) {
        return nullptr;
    }
    return &ctx->batch[static_cast<size_t>(index)];
}

const char* aicore_wevisone_result_answer_at(const aicore_wevisone_ctx* ctx,
                                             int32_t index) {
    const RunResult* res = batch_result(ctx, index);
    return res == nullptr ? nullptr : res->answer.c_str();
}

char* aicore_wevisone_result_json_at(const aicore_wevisone_ctx* ctx,
                                     int32_t index) {
    const RunResult* res = batch_result(ctx, index);
    return res == nullptr ? nullptr : render_result_json(*res);
}

int32_t aicore_wevisone_result_box_count_at(const aicore_wevisone_ctx* ctx,
                                            int32_t index) {
    const RunResult* res = batch_result(ctx, index);
    return res == nullptr ? 0 : static_cast<int32_t>(res->boxes.size());
}

int aicore_wevisone_result_box_at(const aicore_wevisone_ctx* ctx,
                                  int32_t index,
                                  int32_t box_index,
                                  float* out) {
    const RunResult* res = batch_result(ctx, index);
    if (res == nullptr || out == nullptr || box_index < 0 ||
        box_index >= static_cast<int32_t>(res->boxes.size())) {
        return -1;
    }
    const ResultBox& b = res->boxes[static_cast<size_t>(box_index)];
    out[0] = b.x1;
    out[1] = b.y1;
    out[2] = b.x2;
    out[3] = b.y2;
    return 0;
}

int aicore_wevisone_result_has_masks_at(const aicore_wevisone_ctx* ctx,
                                        int32_t index) {
    const RunResult* res = batch_result(ctx, index);
    return res != nullptr && res->masks_decoded ? 1 : 0;
}

int32_t aicore_wevisone_result_matte_count_at(const aicore_wevisone_ctx* ctx,
                                              int32_t index) {
    const RunResult* res = batch_result(ctx, index);
    if (res == nullptr || !res->masks_decoded) return 0;
    return static_cast<int32_t>(res->boxes.size());
}

int aicore_wevisone_result_matte_size_at(const aicore_wevisone_ctx* ctx,
                                         int32_t index,
                                         int32_t matte_index,
                                         int32_t* out_width,
                                         int32_t* out_height) {
    const RunResult* res = batch_result(ctx, index);
    if (res == nullptr || out_width == nullptr || out_height == nullptr ||
        matte_index != 0 || !res->masks_decoded) {
        return -1;
    }
    *out_width = res->matte_width;
    *out_height = res->matte_height;
    return 0;
}

const float* aicore_wevisone_result_matte_data_at(
        const aicore_wevisone_ctx* ctx, int32_t index, int32_t matte_index) {
    const RunResult* res = batch_result(ctx, index);
    if (res == nullptr || matte_index != 0 || !res->masks_decoded) {
        return nullptr;
    }
    return res->matte.data();
}

int aicore_wevisone_result_render_png(const aicore_wevisone_ctx* ctx,
                                      const aicore_image_view* img,
                                      const char* out_png_path) {
    if (ctx == nullptr || img == nullptr || out_png_path == nullptr) return -1;
    const bool ok = render_result_png(
            *img, reinterpret_cast<const float*>(ctx->last.boxes.data()),
            static_cast<int32_t>(ctx->last.boxes.size()),
            ctx->last.masks_decoded ? ctx->last.matte.data() : nullptr,
            ctx->last.matte_width, ctx->last.matte_height, out_png_path);
    return ok ? 0 : -2;
}

int aicore_wevisone_last_pipeline_timings(const aicore_wevisone_ctx* ctx,
                                          aicore_pipeline_timings* out) {
    if (ctx == nullptr || out == nullptr) return -1;
    *out = ctx->last.timings;
    return 0;
}

const char* aicore_wevisone_context_device(const aicore_wevisone_ctx* ctx) {
    return ctx == nullptr || ctx->rt == nullptr
                   ? nullptr
                   : ctx->rt->resolved_device().c_str();
}

int aicore_wevisone_context_threads(const aicore_wevisone_ctx* ctx) {
    return ctx == nullptr || ctx->rt == nullptr ? 0
                                                : ctx->rt->resolved_threads();
}

const char* aicore_wevisone_context_precision(const aicore_wevisone_ctx* ctx) {
    return ctx == nullptr || ctx->rt == nullptr
                   ? nullptr
                   : ctx->rt->resolved_precision().c_str();
}

char* aicore_wevisone_info_json(aicore_wevisone_ctx* ctx) {
    if (ctx == nullptr || ctx->rt == nullptr) return nullptr;
    std::ostringstream json;
    json << "{\"device\": \"" << json_escape(ctx->rt->resolved_device())
         << "\", \"threads\": " << ctx->rt->resolved_threads()
         << ", \"precision\": \"" << ctx->rt->resolved_precision() << "\", "
         << "\"ready\": " << (ctx->rt->is_ready() ? "true" : "false") << "}";
    const std::string s = json.str();
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (out != nullptr) std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

int aicore_wevisone_warmup_backend(aicore_wevisone_ctx* ctx) {
    if (ctx == nullptr || ctx->rt == nullptr) return -1;
    return ctx->rt->warmup_backend() ? 0 : -2;
}

void aicore_wevisone_shutdown(void) {
    // Delegates to aicore_runtime_shutdown(); never destroys live contexts.
    aicore_runtime_shutdown();
}

char* aicore_wevisone_model_cache_dir(void) {
    const std::string dir = aicore::wevisone_model_cache_dir();
    char* out = static_cast<char*>(std::malloc(dir.size() + 1));
    if (out != nullptr) std::memcpy(out, dir.c_str(), dir.size() + 1);
    return out;
}

}  // extern "C"
