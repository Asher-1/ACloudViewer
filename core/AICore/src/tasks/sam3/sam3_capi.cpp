// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// SAM3 C API implementation. Wraps the in-tree sam3.cpp engine (single-file
// SAM 2 / 2.1 / 3 inference on ggml) behind a stable C ABI:
//   - options builder pattern (device / threads / prompt thresholds)
//   - borrowed RGB input (stride-aware, no ownership transfer)
//   - typed segmentation results with borrowed mask views
//   - video tracker with add-instance / refine / reset
//   - published model catalog (cloudViewer_downloads "sam" release)

#include "aicore/sam3_capi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#include "common/aicore_log.hpp"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/model_cache.hpp"
#include "sam3.h"
#include "tasks/sam3/quantize.hpp"

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point t0, Clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// Convert a borrowed stride-aware RGB24 view into the tightly-packed RGB the
// engine preprocess expects. Zero-copy when the stride is already packed.
std::vector<uint8_t> pack_rgb(const uint8_t* rgb,
                              int32_t width,
                              int32_t height,
                              size_t row_stride_bytes) {
    std::vector<uint8_t> packed;
    if (row_stride_bytes == static_cast<size_t>(width) * 3) {
        packed.assign(rgb, rgb + static_cast<size_t>(width) * height * 3);
    } else {
        packed.resize(static_cast<size_t>(width) * height * 3);
        for (int32_t y = 0; y < height; ++y) {
            std::memcpy(packed.data() + static_cast<size_t>(y) * width * 3,
                        rgb + static_cast<size_t>(y) * row_stride_bytes,
                        static_cast<size_t>(width) * 3);
        }
    }
    return packed;
}

sam3_image make_sam3_image(const std::vector<uint8_t>& rgb,
                           int32_t width,
                           int32_t height) {
    sam3_image img;
    img.width = width;
    img.height = height;
    img.channels = 3;
    img.data = rgb;
    return img;
}

// Map an upstream sam3_device enum from the options device string.
sam3_device resolve_device(const std::string& device) {
    if (device == "cpu") return SAM3_DEVICE_CPU;
    if (device == "cuda") return SAM3_DEVICE_CUDA;
    if (device == "vulkan") return SAM3_DEVICE_VULKAN;
    if (device == "metal") return SAM3_DEVICE_METAL;
    return SAM3_DEVICE_AUTO;  // "auto" / NULL / anything else
}

// Shared engine log level for verbose progress (upstream SAM3_LOG_LEVEL=2).
// Kept at INFO unless a caller explicitly raises it via aicore_set_log_level.
constexpr int kSam3LogLevel = 1;

thread_local std::string g_last_load_error;

}  // namespace

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct aicore_sam3_options {
    std::string device = "auto";
    int n_threads = 0;
    int encode_img_size = 0;
    float score_threshold = 0.5f;
    float nms_threshold = 0.1f;
    float assoc_iou_threshold = 0.1f;
    int hotstart_delay = 15;
    int max_keep_alive = 30;
    int recondition_every = 16;
    int fill_hole_area = 16;
};

AICORE_CAPI int aicore_sam3_abi_version(void) { return 1; }

AICORE_CAPI aicore_sam3_options* aicore_sam3_options_new(void) {
    return new (std::nothrow) aicore_sam3_options();
}

AICORE_CAPI void aicore_sam3_options_free(aicore_sam3_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_sam3_options_set_device(aicore_sam3_options* opts,
                                                const char* device) {
    if (!opts) return;
    opts->device = device ? device : "auto";
}

AICORE_CAPI void aicore_sam3_options_set_threads(aicore_sam3_options* opts,
                                                 int n_threads) {
    if (!opts) return;
    opts->n_threads = n_threads;
}

AICORE_CAPI void aicore_sam3_options_set_encode_img_size(
        aicore_sam3_options* opts, int img_size) {
    if (!opts) return;
    opts->encode_img_size = img_size;
}

AICORE_CAPI void aicore_sam3_options_set_score_threshold(
        aicore_sam3_options* opts, float score_threshold) {
    if (!opts) return;
    opts->score_threshold = score_threshold;
}

AICORE_CAPI void aicore_sam3_options_set_nms_threshold(
        aicore_sam3_options* opts, float nms_threshold) {
    if (!opts) return;
    opts->nms_threshold = nms_threshold;
}

AICORE_CAPI void aicore_sam3_options_set_assoc_iou_threshold(
        aicore_sam3_options* opts, float iou_threshold) {
    if (!opts) return;
    opts->assoc_iou_threshold = iou_threshold;
}

AICORE_CAPI void aicore_sam3_options_set_hotstart_delay(
        aicore_sam3_options* opts, int frames) {
    if (!opts) return;
    opts->hotstart_delay = frames;
}

AICORE_CAPI void aicore_sam3_options_set_max_keep_alive(
        aicore_sam3_options* opts, int frames) {
    if (!opts) return;
    opts->max_keep_alive = frames;
}

AICORE_CAPI void aicore_sam3_options_set_recondition_every(
        aicore_sam3_options* opts, int frames) {
    if (!opts) return;
    opts->recondition_every = frames;
}

AICORE_CAPI void aicore_sam3_options_set_fill_hole_area(
        aicore_sam3_options* opts, int area) {
    if (!opts) return;
    opts->fill_hole_area = area;
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct aicore_sam3_ctx {
    std::shared_ptr<sam3_model> model;
    sam3_state_ptr state;
    sam3_params params;
    std::string last_error;
    aicore_sam3_timings timings{};
    int model_type = -1;
    bool visual_only = false;
    std::string backend_name;

    // Prompt/tracker thresholds (kept here, not in sam3_params which only
    // carries load-time settings).
    float score_threshold = 0.5f;
    float nms_threshold = 0.1f;
    float assoc_iou_threshold = 0.1f;
    int hotstart_delay = 15;
    int max_keep_alive = 30;
    int recondition_every = 16;
    int fill_hole_area = 16;

    // Cached encoded image (image mode). The cache identity includes image
    // bytes and the requested neck: same-sized frames and PCS/PVS encodes are
    // not interchangeable.
    bool encoded = false;
    bool encoded_pvs_only = false;
    int32_t encoded_w = 0;
    int32_t encoded_h = 0;
    std::vector<uint8_t> encoded_rgb;
};

AICORE_CAPI aicore_sam3_ctx* aicore_sam3_load_opts(
        const char* gguf_path, const aicore_sam3_options* opts) {
    g_last_load_error.clear();
    if (!gguf_path || !gguf_path[0]) {
        g_last_load_error = "empty model path";
        AICORE_LOG_ERROR("[sam3] ", "load: %s\n", g_last_load_error.c_str());
        return nullptr;
    }

    std::unique_ptr<aicore_sam3_ctx> ctx(new (std::nothrow) aicore_sam3_ctx());
    if (!ctx) {
        g_last_load_error = "context allocation failed";
        return nullptr;
    }

    sam3_params params;
    params.model_path = gguf_path;
    params.n_threads = opts ? opts->n_threads : 0;
    if (params.n_threads <= 0) {
        params.n_threads = 4;
    }
    params.device = resolve_device(opts ? opts->device : std::string("auto"));
    params.encode_img_size = opts ? opts->encode_img_size : 0;

    auto t0 = Clock::now();
    ctx->model = sam3_load_model(params);
    if (!ctx->model) {
        g_last_load_error =
                "model load failed (invalid/incompatible GGUF or backend "
                "allocation failure; see the preceding [sam3] log)";
        AICORE_LOG_ERROR("[sam3] ", "%s\n", g_last_load_error.c_str());
        return nullptr;
    }

    ctx->state = sam3_create_state(*ctx->model, params);
    if (!ctx->state) {
        g_last_load_error = "sam3_create_state failed";
        AICORE_LOG_ERROR("[sam3] ", "%s\n", g_last_load_error.c_str());
        return nullptr;
    }

    ctx->params = params;
    ctx->model_type = static_cast<int>(sam3_get_model_type(*ctx->model));
    ctx->visual_only = sam3_is_visual_only(*ctx->model);
    ctx->backend_name = sam3_backend_name(*ctx->model);
    if (opts) {
        ctx->score_threshold = opts->score_threshold;
        ctx->nms_threshold = opts->nms_threshold;
        ctx->assoc_iou_threshold = opts->assoc_iou_threshold;
        ctx->hotstart_delay = opts->hotstart_delay;
        ctx->max_keep_alive = opts->max_keep_alive;
        ctx->recondition_every = opts->recondition_every;
        ctx->fill_hole_area = opts->fill_hole_area;
    }
    ctx->timings.e2e_ms = elapsed_ms(t0, Clock::now());

    AICORE_LOG_PRINT("[sam3] ", "loaded %s on %s (%d threads, %.0f ms)\n",
                     gguf_path, ctx->backend_name.c_str(), params.n_threads,
                     ctx->timings.e2e_ms);
    return ctx.release();
}

AICORE_CAPI void aicore_sam3_free(aicore_sam3_ctx* ctx) {
    if (!ctx) return;
    // State holds GPU buffers; release before the model is destroyed.
    ctx->state.reset();
    ctx->model.reset();
    delete ctx;
}

AICORE_CAPI int aicore_sam3_is_ready(const aicore_sam3_ctx* ctx) {
    return ctx && ctx->model ? 1 : 0;
}

AICORE_CAPI const char* aicore_sam3_last_error(const aicore_sam3_ctx* ctx) {
    static const char* kEmpty = "";
    return ctx ? ctx->last_error.c_str() : kEmpty;
}

AICORE_CAPI const char* aicore_sam3_last_load_error(void) {
    return g_last_load_error.c_str();
}

AICORE_CAPI int aicore_sam3_set_score_threshold(aicore_sam3_ctx* ctx,
                                                float score_threshold) {
    if (!ctx || !std::isfinite(score_threshold) || score_threshold < 0.0f ||
        score_threshold > 1.0f) {
        return -1;
    }
    ctx->score_threshold = score_threshold;
    return 0;
}

AICORE_CAPI void aicore_sam3_free_buffer(void* p) { std::free(p); }

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_sam3_context_model_type(const aicore_sam3_ctx* ctx) {
    return ctx ? ctx->model_type : -1;
}

AICORE_CAPI int aicore_sam3_context_visual_only(const aicore_sam3_ctx* ctx) {
    return ctx && ctx->visual_only ? 1 : 0;
}

AICORE_CAPI const char* aicore_sam3_context_backend_name(
        const aicore_sam3_ctx* ctx) {
    static const char* kNone = "none";
    return ctx ? ctx->backend_name.c_str() : kNone;
}

AICORE_CAPI int aicore_sam3_context_threads(const aicore_sam3_ctx* ctx) {
    return ctx ? ctx->params.n_threads : 0;
}

// ---------------------------------------------------------------------------
// Image encode
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_sam3_encode_rgb(aicore_sam3_ctx* ctx,
                                       const uint8_t* rgb,
                                       int32_t width,
                                       int32_t height,
                                       size_t row_stride_bytes,
                                       int pvs_only) {
    if (!ctx || !ctx->model || !ctx->state) {
        if (ctx) ctx->last_error = "context not ready";
        return -1;
    }
    if (!rgb || width <= 0 || height <= 0 ||
        row_stride_bytes < static_cast<size_t>(width) * 3) {
        ctx->last_error = "invalid RGB view";
        return -1;
    }

    auto t0 = Clock::now();
    ctx->encoded_rgb = pack_rgb(rgb, width, height, row_stride_bytes);
    const sam3_image img = make_sam3_image(ctx->encoded_rgb, width, height);

    const bool ok =
            pvs_only ? sam3_encode_image_pvs(*ctx->state, *ctx->model, img)
                     : sam3_encode_image(*ctx->state, *ctx->model, img);
    if (!ok) {
        ctx->last_error = "encode failed";
        ctx->encoded = false;
        return -1;
    }
    ctx->encoded = true;
    ctx->encoded_pvs_only = pvs_only != 0;
    ctx->encoded_w = width;
    ctx->encoded_h = height;
    ctx->timings.preprocess_ms = elapsed_ms(t0, Clock::now());
    return 0;
}

AICORE_CAPI int aicore_sam3_has_encoded_image(const aicore_sam3_ctx* ctx) {
    return ctx && ctx->encoded ? 1 : 0;
}

// ---------------------------------------------------------------------------
// Segmentation results
// ---------------------------------------------------------------------------

struct aicore_sam3_seg_result {
    sam3_result result;  // owns detections + masks (allocated by sam3.cpp)
};

static bool result_is_finite_and_well_formed(const sam3_result& result) {
    for (const sam3_detection& det : result.detections) {
        if (!std::isfinite(det.box.x0) || !std::isfinite(det.box.y0) ||
            !std::isfinite(det.box.x1) || !std::isfinite(det.box.y1) ||
            !std::isfinite(det.score) || !std::isfinite(det.iou_score) ||
            !std::isfinite(det.mask.iou_score) ||
            !std::isfinite(det.mask.obj_score) || det.mask.width <= 0 ||
            det.mask.height <= 0 ||
            det.mask.data.size() !=
                    static_cast<size_t>(det.mask.width) * det.mask.height ||
            !std::all_of(det.sam_token.begin(), det.sam_token.end(),
                         [](float v) { return std::isfinite(v); })) {
            return false;
        }
    }
    return true;
}

static bool encoded_rgb_matches(const aicore_sam3_ctx* ctx,
                                const uint8_t* rgb,
                                int32_t width,
                                int32_t height,
                                size_t row_stride_bytes) {
    if (!ctx || !rgb || width <= 0 || height <= 0 ||
        row_stride_bytes < static_cast<size_t>(width) * 3) {
        return false;
    }
    const size_t row_bytes = static_cast<size_t>(width) * 3;
    const size_t expected = row_bytes * static_cast<size_t>(height);
    if (ctx->encoded_rgb.size() != expected) return false;
    for (int32_t y = 0; y < height; ++y) {
        if (std::memcmp(ctx->encoded_rgb.data() +
                                static_cast<size_t>(y) * row_bytes,
                        rgb + static_cast<size_t>(y) * row_stride_bytes,
                        row_bytes) != 0) {
            return false;
        }
    }
    return true;
}

// Reuse an encode only when dimensions, content and neck all match. Width and
// height alone are insufficient for image tabs that switch between same-sized
// inputs or between PCS (full neck) and PVS (visual neck).
static bool ensure_encoded(aicore_sam3_ctx* ctx,
                           const uint8_t* rgb,
                           int32_t width,
                           int32_t height,
                           size_t row_stride_bytes,
                           bool pvs_only) {
    if (ctx->encoded && ctx->encoded_w == width && ctx->encoded_h == height &&
        ctx->encoded_pvs_only == pvs_only &&
        encoded_rgb_matches(ctx, rgb, width, height, row_stride_bytes)) {
        return true;
    }
    return aicore_sam3_encode_rgb(ctx, rgb, width, height, row_stride_bytes,
                                  pvs_only ? 1 : 0) == 0;
}

static aicore_sam3_seg_result* run_segment(aicore_sam3_ctx* ctx,
                                           std::function<sam3_result()> fn,
                                           const char* what) {
    auto t0 = Clock::now();
    sam3_result r = fn();
    if (!result_is_finite_and_well_formed(r)) {
        sam3_free_result(r);
        ctx->last_error = std::string(what) +
                          " produced a non-finite or malformed result";
        return nullptr;
    }
    if (r.detections.empty()) {
        // Empty results are valid (no detections); only report failures via
        // the upstream log. Keep the timing accounting simple.
    }
    std::unique_ptr<aicore_sam3_seg_result> out(
            new (std::nothrow) aicore_sam3_seg_result());
    if (!out) {
        ctx->last_error = "out of memory";
        return nullptr;
    }
    out->result = std::move(r);
    ctx->timings.inference_ms = elapsed_ms(t0, Clock::now());
    ctx->timings.e2e_ms =
            ctx->timings.preprocess_ms + ctx->timings.inference_ms;
    ctx->last_error.clear();
    return out.release();
}

AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_segment_pcs_rgb(
        aicore_sam3_ctx* ctx,
        const aicore_sam3_pcs_prompt* prompt,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes) {
    if (!ctx || !ctx->model || !ctx->state) {
        if (ctx) ctx->last_error = "context not ready";
        return nullptr;
    }
    if (!prompt) {
        ctx->last_error =
                "PCS requires a text prompt or at least one exemplar box";
        return nullptr;
    }
    // Mirror upstream examples/main_image.cpp: the text prompt is optional
    // when exemplar boxes are provided (and vice versa). Only reject when
    // both are empty — sam3_segment_pcs handles a missing text prompt by
    // falling back to SOT/EOT-only tokens.
    const bool no_text = !prompt->text || !prompt->text[0];
    const bool no_exemplars =
            prompt->n_pos_exemplars <= 0 && prompt->n_neg_exemplars <= 0;
    if (no_text && no_exemplars) {
        ctx->last_error =
                "PCS requires a text prompt or at least one exemplar box";
        return nullptr;
    }
    if (ctx->visual_only) {
        ctx->last_error = "model is visual-only; PCS text mode unavailable";
        return nullptr;
    }
    if (!ensure_encoded(ctx, rgb, width, height, row_stride_bytes, false)) {
        return nullptr;
    }

    sam3_pcs_params p;
    p.text_prompt = prompt->text;
    p.score_threshold = prompt->score_threshold > 0 ? prompt->score_threshold
                                                    : ctx->score_threshold;
    p.nms_threshold = prompt->nms_threshold > 0 ? prompt->nms_threshold
                                                : ctx->nms_threshold;
    if (prompt->pos_exemplars && prompt->n_pos_exemplars > 0) {
        p.pos_exemplars.reserve(prompt->n_pos_exemplars);
        for (int i = 0; i < prompt->n_pos_exemplars; ++i) {
            p.pos_exemplars.push_back(
                    {prompt->pos_exemplars[i].x0, prompt->pos_exemplars[i].y0,
                     prompt->pos_exemplars[i].x1, prompt->pos_exemplars[i].y1});
        }
    }
    if (prompt->neg_exemplars && prompt->n_neg_exemplars > 0) {
        p.neg_exemplars.reserve(prompt->n_neg_exemplars);
        for (int i = 0; i < prompt->n_neg_exemplars; ++i) {
            p.neg_exemplars.push_back(
                    {prompt->neg_exemplars[i].x0, prompt->neg_exemplars[i].y0,
                     prompt->neg_exemplars[i].x1, prompt->neg_exemplars[i].y1});
        }
    }

    return run_segment(
            ctx,
            [&]() { return sam3_segment_pcs(*ctx->state, *ctx->model, p); },
            "pcs");
}

AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_segment_pvs_rgb(
        aicore_sam3_ctx* ctx,
        const aicore_sam3_pvs_prompt* prompt,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes) {
    if (!ctx || !ctx->model || !ctx->state) {
        if (ctx) ctx->last_error = "context not ready";
        return nullptr;
    }
    if (!prompt) {
        ctx->last_error = "null PVS prompt";
        return nullptr;
    }
    if (!ensure_encoded(ctx, rgb, width, height, row_stride_bytes, true)) {
        return nullptr;
    }

    sam3_pvs_params p;
    if (prompt->pos_points && prompt->n_pos_points > 0) {
        p.pos_points.reserve(prompt->n_pos_points);
        for (int i = 0; i < prompt->n_pos_points; ++i) {
            p.pos_points.push_back(
                    {prompt->pos_points[i].x, prompt->pos_points[i].y});
        }
    }
    if (prompt->neg_points && prompt->n_neg_points > 0) {
        p.neg_points.reserve(prompt->n_neg_points);
        for (int i = 0; i < prompt->n_neg_points; ++i) {
            p.neg_points.push_back(
                    {prompt->neg_points[i].x, prompt->neg_points[i].y});
        }
    }
    p.box = {prompt->box.x0, prompt->box.y0, prompt->box.x1, prompt->box.y1};
    p.use_box = prompt->use_box != 0;
    p.multimask = prompt->multimask != 0;

    return run_segment(
            ctx,
            [&]() { return sam3_segment_pvs(*ctx->state, *ctx->model, p); },
            "pvs");
}

AICORE_CAPI int aicore_sam3_seg_det_count(const aicore_sam3_seg_result* res) {
    return res ? static_cast<int>(res->result.detections.size()) : 0;
}

AICORE_CAPI aicore_sam3_box
aicore_sam3_seg_det_box_at(const aicore_sam3_seg_result* res, int index) {
    if (!res || index < 0 || index >= aicore_sam3_seg_det_count(res)) {
        return {0, 0, 0, 0};
    }
    const sam3_box& b = res->result.detections[index].box;
    return {b.x0, b.y0, b.x1, b.y1};
}

AICORE_CAPI float aicore_sam3_seg_det_score_at(
        const aicore_sam3_seg_result* res, int index) {
    if (!res || index < 0 || index >= aicore_sam3_seg_det_count(res)) {
        return 0.0f;
    }
    return res->result.detections[index].score;
}

AICORE_CAPI float aicore_sam3_seg_det_iou_at(const aicore_sam3_seg_result* res,
                                             int index) {
    if (!res || index < 0 || index >= aicore_sam3_seg_det_count(res)) {
        return 0.0f;
    }
    return res->result.detections[index].iou_score;
}

AICORE_CAPI int aicore_sam3_seg_det_instance_id_at(
        const aicore_sam3_seg_result* res, int index) {
    if (!res || index < 0 || index >= aicore_sam3_seg_det_count(res)) {
        return -1;
    }
    return res->result.detections[index].instance_id;
}

AICORE_CAPI aicore_sam3_plane_view
aicore_sam3_seg_mask_at(const aicore_sam3_seg_result* res, int index) {
    if (!res || index < 0 || index >= aicore_sam3_seg_det_count(res)) {
        return {nullptr, 0, 0, 0};
    }
    const sam3_mask& m = res->result.detections[index].mask;
    return {m.data.data(), m.width, m.height, static_cast<size_t>(m.width)};
}

AICORE_CAPI void aicore_sam3_seg_result_free(aicore_sam3_seg_result* res) {
    if (!res) return;
    // Drain vectors inside sam3.cpp (cross-module allocator pairing).
    sam3_free_result(res->result);
    delete res;
}

// ---------------------------------------------------------------------------
// Tracker
// ---------------------------------------------------------------------------

struct aicore_sam3_tracker_ctx {
    std::shared_ptr<sam3_model> model;
    sam3_tracker_ptr tracker;
    sam3_state_ptr state;
    sam3_params params;
    std::string last_error;
    aicore_sam3_timings timings{};
    bool visual_only = false;
};

AICORE_CAPI aicore_sam3_tracker_ctx* aicore_sam3_tracker_create(
        aicore_sam3_ctx* ctx) {
    if (!ctx || !ctx->model) {
        return nullptr;
    }
    std::unique_ptr<aicore_sam3_tracker_ctx> t(
            new (std::nothrow) aicore_sam3_tracker_ctx());
    if (!t) {
        return nullptr;
    }
    t->model = ctx->model;
    t->params = ctx->params;
    t->visual_only = ctx->visual_only;

    if (ctx->visual_only) {
        sam3_visual_track_params vp;
        vp.assoc_iou_threshold = ctx->assoc_iou_threshold;
        vp.max_keep_alive = ctx->max_keep_alive;
        vp.recondition_every = ctx->recondition_every;
        vp.fill_hole_area = ctx->fill_hole_area;
        t->tracker = sam3_create_visual_tracker(*t->model, vp);
    } else {
        sam3_video_params vp;
        vp.score_threshold = ctx->score_threshold;
        vp.nms_threshold = ctx->nms_threshold;
        vp.assoc_iou_threshold = ctx->assoc_iou_threshold;
        vp.hotstart_delay = ctx->hotstart_delay;
        vp.max_keep_alive = ctx->max_keep_alive;
        vp.recondition_every = ctx->recondition_every;
        vp.fill_hole_area = ctx->fill_hole_area;
        t->tracker = sam3_create_tracker(*t->model, vp);
    }
    if (!t->tracker) {
        t->last_error = "sam3_create_tracker failed";
        ctx->last_error = t->last_error;
        return nullptr;
    }
    t->state = sam3_create_state(*t->model, t->params);
    if (!t->state) {
        t->last_error = "sam3_create_state failed";
        ctx->last_error = t->last_error;
        return nullptr;
    }
    return t.release();
}

AICORE_CAPI const char* aicore_sam3_tracker_last_error(
        const aicore_sam3_tracker_ctx* tracker) {
    static const char* kEmpty = "";
    return tracker ? tracker->last_error.c_str() : kEmpty;
}

AICORE_CAPI void aicore_sam3_tracker_set_text_prompt(
        aicore_sam3_tracker_ctx* tracker, const char* text) {
    if (!tracker || !tracker->tracker || tracker->visual_only) return;
    sam3_tracker_set_text_prompt(*tracker->tracker,
                                 text ? text : std::string());
}

AICORE_CAPI void aicore_sam3_tracker_free(aicore_sam3_tracker_ctx* tracker) {
    if (!tracker) return;
    tracker->state.reset();
    tracker->tracker.reset();
    tracker->model.reset();
    delete tracker;
}

static aicore_sam3_seg_result* run_tracker_frame(
        aicore_sam3_tracker_ctx* tracker,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes,
        bool propagate_only) {
    if (!tracker || !tracker->tracker || !tracker->state || !tracker->model) {
        if (tracker) tracker->last_error = "tracker not ready";
        return nullptr;
    }
    if (!rgb || width <= 0 || height <= 0 ||
        row_stride_bytes < static_cast<size_t>(width) * 3) {
        tracker->last_error = "invalid RGB view";
        return nullptr;
    }

    auto t0 = Clock::now();
    const std::vector<uint8_t> packed =
            pack_rgb(rgb, width, height, row_stride_bytes);
    const sam3_image frame = make_sam3_image(packed, width, height);

    auto t1 = Clock::now();
    sam3_result r =
            propagate_only
                    ? sam3_propagate_frame(*tracker->tracker, *tracker->state,
                                           *tracker->model, frame)
                    : sam3_track_frame(*tracker->tracker, *tracker->state,
                                       *tracker->model, frame);
    auto t2 = Clock::now();

    if (!result_is_finite_and_well_formed(r)) {
        sam3_free_result(r);
        tracker->last_error =
                "tracker produced a non-finite or malformed result";
        return nullptr;
    }

    std::unique_ptr<aicore_sam3_seg_result> out(
            new (std::nothrow) aicore_sam3_seg_result());
    if (!out) {
        tracker->last_error = "out of memory";
        return nullptr;
    }
    out->result = std::move(r);
    tracker->timings.preprocess_ms = elapsed_ms(t0, t1);
    tracker->timings.inference_ms = elapsed_ms(t1, t2);
    tracker->timings.e2e_ms = elapsed_ms(t0, t2);
    tracker->last_error.clear();
    return out.release();
}

AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_track_frame(
        aicore_sam3_tracker_ctx* tracker,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes) {
    return run_tracker_frame(tracker, rgb, width, height, row_stride_bytes,
                             false);
}

AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_propagate_frame(
        aicore_sam3_tracker_ctx* tracker,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        size_t row_stride_bytes) {
    return run_tracker_frame(tracker, rgb, width, height, row_stride_bytes,
                             true);
}

// Convert a C PVS prompt into the engine's sam3_pvs_params. Returns false on
// a null prompt (all-empty prompts are allowed: they mirror the upstream
// behavior of an empty prompt set).
static bool fill_pvs_params(sam3_pvs_params& p,
                            const aicore_sam3_pvs_prompt* prompt) {
    if (!prompt) return false;
    if (prompt->pos_points && prompt->n_pos_points > 0) {
        p.pos_points.reserve(prompt->n_pos_points);
        for (int i = 0; i < prompt->n_pos_points; ++i) {
            p.pos_points.push_back(
                    {prompt->pos_points[i].x, prompt->pos_points[i].y});
        }
    }
    if (prompt->neg_points && prompt->n_neg_points > 0) {
        p.neg_points.reserve(prompt->n_neg_points);
        for (int i = 0; i < prompt->n_neg_points; ++i) {
            p.neg_points.push_back(
                    {prompt->neg_points[i].x, prompt->neg_points[i].y});
        }
    }
    p.box = {prompt->box.x0, prompt->box.y0, prompt->box.x1, prompt->box.y1};
    p.use_box = prompt->use_box != 0;
    p.multimask = prompt->multimask != 0;
    return true;
}

AICORE_CAPI int aicore_sam3_tracker_add_instance(
        aicore_sam3_tracker_ctx* tracker,
        const aicore_sam3_pvs_prompt* prompt) {
    if (!tracker || !tracker->tracker || !tracker->state || !tracker->model) {
        if (tracker) tracker->last_error = "tracker not ready";
        return -1;
    }
    sam3_pvs_params p;
    if (!fill_pvs_params(p, prompt)) {
        tracker->last_error = "null PVS prompt";
        return -1;
    }

    const int id = sam3_tracker_add_instance(*tracker->tracker, *tracker->state,
                                             *tracker->model, p);
    if (id < 0) {
        tracker->last_error = "add_instance failed";
    } else {
        tracker->last_error.clear();
    }
    return id;
}

AICORE_CAPI aicore_sam3_seg_result* aicore_sam3_tracker_segment_pvs(
        aicore_sam3_tracker_ctx* tracker,
        const aicore_sam3_pvs_prompt* prompt) {
    if (!tracker || !tracker->tracker || !tracker->state || !tracker->model) {
        if (tracker) tracker->last_error = "tracker not ready";
        return nullptr;
    }
    sam3_pvs_params p;
    if (!fill_pvs_params(p, prompt)) {
        tracker->last_error = "null PVS prompt";
        return nullptr;
    }
    if (p.pos_points.empty() && !p.use_box) {
        tracker->last_error =
                "no prompts provided (need at least one point or box)";
        return nullptr;
    }

    auto t0 = Clock::now();
    sam3_result r = sam3_segment_pvs(*tracker->state, *tracker->model, p);
    auto t1 = Clock::now();
    if (!result_is_finite_and_well_formed(r)) {
        sam3_free_result(r);
        tracker->last_error =
                "tracker PVS produced a non-finite or malformed result";
        return nullptr;
    }
    // sam3_segment_pvs is safe on an unencoded state (logs + empty result);
    // an empty detection list is a valid outcome here (e.g. a prompt that
    // matches nothing), so we hand it back as a regular result.
    std::unique_ptr<aicore_sam3_seg_result> out(
            new (std::nothrow) aicore_sam3_seg_result());
    if (!out) {
        tracker->last_error = "out of memory";
        return nullptr;
    }
    out->result = std::move(r);
    tracker->timings.preprocess_ms = 0.0;
    tracker->timings.inference_ms = elapsed_ms(t0, t1);
    tracker->timings.e2e_ms = elapsed_ms(t0, t1);
    tracker->last_error.clear();
    return out.release();
}

AICORE_CAPI int aicore_sam3_tracker_add_instance_from_mask(
        aicore_sam3_tracker_ctx* tracker, const aicore_sam3_plane_view* mask) {
    if (!tracker || !tracker->tracker || !tracker->state || !tracker->model) {
        if (tracker) tracker->last_error = "tracker not ready";
        return -1;
    }
    if (!mask || !mask->data || mask->width <= 0 || mask->height <= 0 ||
        mask->row_stride_bytes < static_cast<size_t>(mask->width)) {
        tracker->last_error = "invalid mask view";
        return -1;
    }
    sam3_mask m;
    m.width = mask->width;
    m.height = mask->height;
    // Unpack the (possibly padded) mask rows into a tightly packed buffer.
    m.data.resize(static_cast<size_t>(mask->width) * mask->height);
    const uint8_t* src = static_cast<const uint8_t*>(mask->data);
    for (int32_t y = 0; y < mask->height; ++y) {
        std::memcpy(m.data.data() + static_cast<size_t>(y) * mask->width,
                    src + static_cast<size_t>(y) * mask->row_stride_bytes,
                    static_cast<size_t>(mask->width));
    }
    const int id = sam3_tracker_add_instance_from_mask(
            *tracker->tracker, *tracker->state, *tracker->model, m);
    if (id < 0) {
        tracker->last_error = "add_instance_from_mask failed";
    } else {
        tracker->last_error.clear();
    }
    return id;
}

AICORE_CAPI int aicore_sam3_refine_instance(aicore_sam3_tracker_ctx* tracker,
                                            int instance_id,
                                            const aicore_sam3_point* pos_points,
                                            int n_pos_points,
                                            const aicore_sam3_point* neg_points,
                                            int n_neg_points) {
    if (!tracker || !tracker->tracker || !tracker->state || !tracker->model) {
        if (tracker) tracker->last_error = "tracker not ready";
        return -1;
    }
    std::vector<sam3_point> pos, neg;
    if (pos_points && n_pos_points > 0) {
        pos.reserve(n_pos_points);
        for (int i = 0; i < n_pos_points; ++i) {
            pos.push_back({pos_points[i].x, pos_points[i].y});
        }
    }
    if (neg_points && n_neg_points > 0) {
        neg.reserve(n_neg_points);
        for (int i = 0; i < n_neg_points; ++i) {
            neg.push_back({neg_points[i].x, neg_points[i].y});
        }
    }
    const bool ok =
            sam3_refine_instance(*tracker->tracker, *tracker->state,
                                 *tracker->model, instance_id, pos, neg);
    if (!ok) {
        tracker->last_error = "refine_instance failed";
        return -1;
    }
    tracker->last_error.clear();
    return 0;
}

AICORE_CAPI int aicore_sam3_tracker_frame_index(
        const aicore_sam3_tracker_ctx* tracker) {
    return tracker && tracker->tracker
                   ? sam3_tracker_frame_index(*tracker->tracker)
                   : -1;
}

AICORE_CAPI void aicore_sam3_tracker_reset(aicore_sam3_tracker_ctx* tracker) {
    if (tracker && tracker->tracker) {
        sam3_tracker_reset(*tracker->tracker);
    }
}

// ---------------------------------------------------------------------------
// Timings
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_sam3_last_timings(const aicore_sam3_ctx* ctx,
                                         aicore_sam3_timings* out_timings) {
    if (!ctx || !out_timings) {
        return -1;
    }
    *out_timings = ctx->timings;
    return ctx->timings.e2e_ms > 0.0 ? 0 : -1;
}

namespace {

int copy_pipeline_timings(const aicore_sam3_timings& source,
                          aicore_pipeline_timings* destination) {
    if (destination == nullptr || source.e2e_ms <= 0.0) return -1;
    *destination = aicore_pipeline_timings{
            AICORE_PIPELINE_TIMINGS_ABI_VERSION,
            AICORE_TIMING_PREPROCESS | AICORE_TIMING_INFERENCE |
                    AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E,
            source.preprocess_ms,
            source.inference_ms,
            source.postprocess_ms,
            0.0,
            source.e2e_ms};
    return 0;
}

}  // namespace

AICORE_CAPI int aicore_sam3_last_pipeline_timings(
        const aicore_sam3_ctx* ctx, aicore_pipeline_timings* out_timings) {
    return ctx == nullptr ? -1
                          : copy_pipeline_timings(ctx->timings, out_timings);
}

AICORE_CAPI int aicore_sam3_tracker_last_timings(
        const aicore_sam3_tracker_ctx* tracker,
        aicore_sam3_timings* out_timings) {
    if (!tracker || !out_timings) return -1;
    *out_timings = tracker->timings;
    return tracker->timings.e2e_ms > 0.0 ? 0 : -1;
}

AICORE_CAPI int aicore_sam3_tracker_last_pipeline_timings(
        const aicore_sam3_tracker_ctx* tracker,
        aicore_pipeline_timings* out_timings) {
    return tracker == nullptr
                   ? -1
                   : copy_pipeline_timings(tracker->timings, out_timings);
}

// ---------------------------------------------------------------------------
// Model catalog (cloudViewer_downloads "sam" release)
// ---------------------------------------------------------------------------

namespace {

struct ModelEntry {
    const char* filename;
    const char* family;
    int64_t size_bytes;
    const char* quant_note;
};

// Published assets of the "sam" release, verified against the GitHub Release
// API (39 models; sam3-f32 intentionally absent — too large to publish).
constexpr const char* kSamDownloadBase =
        "https://github.com/Asher-1/cloudViewer_downloads/releases/download/"
        "sam/";

constexpr ModelEntry kModels[] = {
        // SAM 3 (full: ViT + text detector + tracker)
        {"sam3-f16.gguf", "sam3", 1837924096LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam3-q8_0.gguf", "sam3", 1099442368LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam3-q4_1.gguf", "sam3", 755755328LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam3-q4_0.gguf", "sam3", 706657216LL,
         "Q4_0 \xe2\x80\x94 smallest SAM3 quant"},
        // SAM 3 visual-only (no text encoder)
        {"sam3-visual-f16.gguf", "sam3-visual", 945529696LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam3-visual-q8_0.gguf", "sam3-visual", 517085888LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam3-visual-q4_1.gguf", "sam3-visual", 317650048LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam3-visual-q4_0.gguf", "sam3-visual", 289159232LL,
         "Q4_0 \xe2\x80\x94 smallest SAM3-visual quant"},
        // SAM 2.1 (Hiera backbone, visual-only)
        {"sam2.1_hiera_large_f16.gguf", "sam2.1", 450932736LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_large_f32.gguf", "sam2.1", 897848576LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_large_q8_0.gguf", "sam2.1", 241243168LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_large_q4_1.gguf", "sam2.1", 143892640LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_large_q4_0.gguf", "sam2.1", 129985440LL,
         "Q4_0 \xe2\x80\x94 smallest large quant"},
        {"sam2.1_hiera_base_plus_f16.gguf", "sam2.1", 163305952LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_base_plus_f32.gguf", "sam2.1", 323444448LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_base_plus_q8_0.gguf", "sam2.1", 87743744LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_base_plus_q4_1.gguf", "sam2.1", 52985984LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_base_plus_q4_0.gguf", "sam2.1", 48020608LL,
         "Q4_0 \xe2\x80\x94 smallest base-plus quant"},
        {"sam2.1_hiera_small_f16.gguf", "sam2.1", 93561600LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_small_f32.gguf", "sam2.1", 184279040LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_small_q8_0.gguf", "sam2.1", 50200672LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_small_q4_1.gguf", "sam2.1", 30470176LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_small_q4_0.gguf", "sam2.1", 27651552LL,
         "Q4_0 \xe2\x80\x94 smallest small quant"},
        {"sam2.1_hiera_tiny_f16.gguf", "sam2.1", 79322912LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_tiny_f32.gguf", "sam2.1", 155884576LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_tiny_q8_0.gguf", "sam2.1", 42597504LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_tiny_q4_1.gguf", "sam2.1", 25963584LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_tiny_q4_0.gguf", "sam2.1", 23587328LL,
         "Q4_0 \xe2\x80\x94 smallest SAM2.1 quant (22 MB)"},
        // SAM 2 (Hiera backbone, visual-only)
        {"sam2_hiera_large_f16.gguf", "sam2", 450866528LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2_hiera_base_plus_f16.gguf", "sam2", 163239712LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2_hiera_base_plus_f32.gguf", "sam2", 323378208LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2_hiera_base_plus_q8_0.gguf", "sam2", 87725632LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2_hiera_base_plus_q4_1.gguf", "sam2", 52975040LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2_hiera_base_plus_q4_0.gguf", "sam2", 48010688LL,
         "Q4_0 \xe2\x80\x94 smallest base-plus quant"},
        {"sam2_hiera_tiny_f16.gguf", "sam2", 79256672LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2_hiera_tiny_f32.gguf", "sam2", 155818336LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2_hiera_tiny_q8_0.gguf", "sam2", 42579392LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2_hiera_tiny_q4_1.gguf", "sam2", 25952640LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2_hiera_tiny_q4_0.gguf", "sam2", 23577408LL,
         "Q4_0 \xe2\x80\x94 smallest SAM2 quant (22 MB)"},
};

constexpr int kModelCount =
        static_cast<int>(sizeof(kModels) / sizeof(kModels[0]));

const char* familyDisplayName(const char* family) {
    if (std::strcmp(family, "sam3") == 0) return "SAM 3 (text + tracking)";
    if (std::strcmp(family, "sam3-visual") == 0) return "SAM 3 visual-only";
    if (std::strcmp(family, "sam2.1") == 0) return "SAM 2.1 Hiera";
    return "SAM 2 Hiera";
}

struct ModelCatalogStore {
    std::vector<std::string> urls;
    std::vector<aicore_sam3_model_entry> entries;

    ModelCatalogStore() {
        urls.reserve(kModelCount);
        entries.reserve(kModelCount);
        for (int i = 0; i < kModelCount; ++i) {
            const ModelEntry& model = kModels[i];
            urls.emplace_back(std::string(kSamDownloadBase) + model.filename);

            aicore_sam3_model_entry entry{};
            entry.filename = model.filename;
            entry.download_url = urls.back().c_str();
            entry.display_name = familyDisplayName(model.family);
            entry.quant_note = model.quant_note;
            entry.model_family = model.family;
            entry.size_bytes = model.size_bytes;
            entry.visual_only = std::strcmp(model.family, "sam3") != 0 ? 1 : 0;
            entries.push_back(entry);
        }
    }
};

const ModelCatalogStore& modelCatalogStore() {
    static const ModelCatalogStore store;
    return store;
}

}  // namespace

AICORE_CAPI int aicore_sam3_model_count(void) { return kModelCount; }

AICORE_CAPI const aicore_sam3_model_entry* aicore_sam3_model_at(int index) {
    if (index < 0 || index >= kModelCount) {
        return nullptr;
    }
    return &modelCatalogStore().entries[index];
}

AICORE_CAPI const aicore_sam3_model_entry* aicore_sam3_model_by_filename(
        const char* filename) {
    if (!filename) {
        return nullptr;
    }
    for (int i = 0; i < kModelCount; ++i) {
        const aicore_sam3_model_entry* e = aicore_sam3_model_at(i);
        if (std::strcmp(e->filename, filename) == 0) {
            return e;
        }
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_sam3_model_download_base(void) {
    return kSamDownloadBase;
}

// ---------------------------------------------------------------------------
// Process-wide helpers
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_sam3_benchmark(aicore_sam3_ctx* ctx,
                                      int32_t img_width,
                                      int32_t img_height,
                                      int n_warmup,
                                      int n_iter,
                                      aicore_sam3_timings* out_avg) {
    if (!ctx || !out_avg || img_width <= 0 || img_height <= 0) return -1;
    if (n_warmup < 1) n_warmup = 1;
    if (n_iter < 1) n_iter = 1;

    // Constant-gray fake frame so results do not depend on image content.
    const size_t row_stride = static_cast<size_t>(img_width) * 3;
    const size_t frame_bytes = static_cast<size_t>(img_height) * row_stride;
    std::vector<uint8_t> frame(frame_bytes, 128);  // mid-gray

    // Point prompt at the image center (mirrors the upstream benchmark's
    // object-tracking point), so every iteration runs the full PVS pipeline
    // (encode + prompt encoder + mask decoder).
    aicore_sam3_point center{static_cast<float>(img_width) * 0.5f,
                             static_cast<float>(img_height) * 0.5f};
    aicore_sam3_pvs_prompt prompt{};
    prompt.pos_points = &center;
    prompt.n_pos_points = 1;

    // Warm-up iterations (no timings recorded).
    for (int i = 0; i < n_warmup; ++i) {
        aicore_sam3_seg_result* r = aicore_sam3_segment_pvs_rgb(
                ctx, &prompt, frame.data(), img_width, img_height, row_stride);
        aicore_sam3_seg_result_free(r);
    }

    // Timed iterations.
    double sum_pre = 0.0, sum_inf = 0.0, sum_post = 0.0, sum_e2e = 0.0;
    for (int i = 0; i < n_iter; ++i) {
        aicore_sam3_seg_result* r = aicore_sam3_segment_pvs_rgb(
                ctx, &prompt, frame.data(), img_width, img_height, row_stride);
        aicore_sam3_timings t{};
        aicore_sam3_last_timings(ctx, &t);
        sum_pre += t.preprocess_ms;
        sum_inf += t.inference_ms;
        sum_post += t.postprocess_ms;
        sum_e2e += t.e2e_ms;
        aicore_sam3_seg_result_free(r);
    }

    out_avg->preprocess_ms = sum_pre / n_iter;
    out_avg->inference_ms = sum_inf / n_iter;
    out_avg->postprocess_ms = sum_post / n_iter;
    out_avg->e2e_ms = sum_e2e / n_iter;
    return 0;
}

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_sam3_profile_encoder(aicore_sam3_ctx* ctx,
                                            int n_warmup,
                                            int n_iter,
                                            aicore_sam3_profile_entry* out,
                                            int max_entries,
                                            int* n_entries_out) {
    if (!ctx || !ctx->model) return -1;
    if (n_warmup < 1) n_warmup = 1;
    if (n_iter < 1) n_iter = 1;
    if (n_entries_out) *n_entries_out = 0;

    const int n_threads = ctx->params.n_threads > 0 ? ctx->params.n_threads : 4;
    int written = 0;
    auto emit = [&](int kind, int index, int stage, double ms) {
        if (out && written < max_entries) {
            out[written].kind = kind;
            out[written].index = index;
            out[written].stage = stage;
            out[written].avg_ms = ms;
        }
        ++written;
    };

    // Infer network geometry from a few known weight tensors.
    sam3_tensor_info ti;
    int patch = 14, E = 0, grid = 0;
    if (sam3_get_model_tensor_info(*ctx->model, "vit.patch_embed.proj.weight",
                                   ti)) {
        patch = (int)ti.ne[0];
        E = (int)ti.ne[3];
    }
    // Full-image feature grid: the largest RoPE table among the blocks is the
    // global-attention one, sized to the full token count (img_size/patch)^2.
    // (Window-attention blocks carry a window-sized table instead.)
    {
        int64_t max_n = 0;
        for (int b = 0; b < 64; ++b) {
            const std::string name =
                    "vit.blocks." + std::to_string(b) + ".attn.freqs_cis";
            if (!sam3_get_model_tensor_info(*ctx->model, name, ti)) break;
            max_n = std::max(max_n, ti.ne[2]);
        }
        if (max_n > 0) {
            const int64_t g = (int64_t)std::sqrt((double)max_n);
            if (g * g == max_n && g > 0) grid = (int)g;
        }
    }
    if (grid <= 0) {
        // Fallback: SAM3 pretrained pos_embed grid (tiled 3x at runtime, see
        // sam3_register_tensors).
        if (sam3_get_model_tensor_info(*ctx->model, "vit.pos_embed", ti)) {
            grid = (int)ti.ne[1] * 3;
        }
    }
    if (E <= 0 || grid <= 0) {
        // SAM2 Hiera models have no ViT patch_embed/pos_embed tensors; the
        // profile sub-graphs are SAM3-only.
        ctx->last_error =
                "profile: model has no ViT tensors (SAM3 profile is SAM3-only)";
        return -1;
    }
    AICORE_LOG_PRINT("[sam3] ", "profile: patch=%d E=%d grid=%d img=%d\n",
                     patch, E, grid, grid * patch);

    const int img_size = grid * patch;

    // ── Prefix stages (patch embed -> ln_pre), chained ───────────────────
    int64_t img_ne[4] = {img_size, img_size, 3, 1};
    std::vector<float> img((size_t)img_size * img_size * 3, 0.001f);
    for (int s = (int)SAM3_VIT_PREFIX_STAGE_PATCH_EMBED;
         s <= (int)SAM3_VIT_PREFIX_STAGE_LN_PRE; ++s) {
        std::vector<float> out_data;
        int64_t out_ne[4] = {0, 0, 0, 0};
        if (!sam3_test_run_vit_prefix_stage(
                    *ctx->model, (sam3_vit_prefix_stage)s, img.data(), img_ne,
                    out_data, out_ne, n_threads)) {
            continue;  // backend does not support this sub-stage
        }
        for (int it = 0; it < n_warmup; ++it) {
            sam3_test_run_vit_prefix_stage(*ctx->model,
                                           (sam3_vit_prefix_stage)s, img.data(),
                                           img_ne, out_data, out_ne, n_threads);
        }
        auto t0 = Clock::now();
        for (int it = 0; it < n_iter; ++it) {
            sam3_test_run_vit_prefix_stage(*ctx->model,
                                           (sam3_vit_prefix_stage)s, img.data(),
                                           img_ne, out_data, out_ne, n_threads);
        }
        emit(AICORE_SAM3_PROFILE_PREFIX, s, s,
             elapsed_ms(t0, Clock::now()) / n_iter);

        // Chain the output as the next stage's input where shapes allow.
        std::copy(std::begin(out_ne), std::end(out_ne), std::begin(img_ne));
        img.swap(out_data);
        img.resize((size_t)img_ne[0] * img_ne[1] * img_ne[2] * img_ne[3], 0.0f);
    }

    // ── Block stages, chained per block ──────────────────────────────────
    std::vector<float> feat((size_t)E * grid * grid, 0.001f);
    int64_t feat_ne[4] = {E, grid, grid, 1};
    for (int b = 0; b < 128; ++b) {  // probe bound; stops when stage 0 fails
        // Global-attention blocks (RoPE table sized to the full token grid)
        // expect un-windowed full-image features, which the chained
        // windowed sub-graphs cannot produce — skip them.
        {
            const std::string name =
                    "vit.blocks." + std::to_string(b) + ".attn.freqs_cis";
            if (sam3_get_model_tensor_info(*ctx->model, name, ti) &&
                ti.ne[2] == (int64_t)grid * grid) {
                continue;
            }
        }
        std::vector<float> x = feat;
        int64_t ne[4];
        std::copy(std::begin(feat_ne), std::end(feat_ne), ne);
        bool any = false;
        for (int s = 0; s <= (int)SAM3_VIT_BLOCK_STAGE_MLP; ++s) {
            std::vector<float> out_data;
            int64_t out_ne[4] = {0, 0, 0, 0};
            if (!sam3_test_run_vit_block_stage(
                        *ctx->model, b, (sam3_vit_block_stage)s, x.data(), ne,
                        out_data, out_ne, n_threads)) {
                continue;  // invalid block or unsupported stage
            }
            any = true;
            for (int it = 0; it < n_warmup; ++it) {
                sam3_test_run_vit_block_stage(*ctx->model, b,
                                              (sam3_vit_block_stage)s, x.data(),
                                              ne, out_data, out_ne, n_threads);
            }
            auto t0 = Clock::now();
            for (int it = 0; it < n_iter; ++it) {
                sam3_test_run_vit_block_stage(*ctx->model, b,
                                              (sam3_vit_block_stage)s, x.data(),
                                              ne, out_data, out_ne, n_threads);
            }
            emit(AICORE_SAM3_PROFILE_BLOCK, b, s,
                 elapsed_ms(t0, Clock::now()) / n_iter);

            std::copy(std::begin(out_ne), std::end(out_ne), std::begin(ne));
            x.swap(out_data);
            x.resize((size_t)ne[0] * ne[1] * ne[2] * ne[3], 0.0f);
        }
        if (!any) break;  // past the last block
    }

    if (n_entries_out) *n_entries_out = written;
    return 0;
}

AICORE_CAPI int aicore_sam3_quantize_gguf(const char* input_gguf,
                                          const char* output_gguf,
                                          const char* type_name) {
    if (!input_gguf || !output_gguf || !type_name) return -1;
    return aicore::sam3::quantize_gguf(input_gguf, output_gguf, type_name) ? 0
                                                                           : -1;
}

AICORE_CAPI int aicore_sam3_warmup_backend(const char* device) {
    // Probe the requested device and clear any stale CUDA error state, so
    // plugin startup can validate the backend before the first load.
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_sam3_shutdown(void) { aicore_runtime_shutdown(); }

AICORE_CAPI char* aicore_sam3_model_cache_dir(void) {
    return aicore::capi::dup_cstr(aicore::sam3_model_cache_dir());
}
