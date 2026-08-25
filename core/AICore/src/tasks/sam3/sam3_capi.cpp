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
#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "common/aicore_log.hpp"
#include "common/ggml_backend_registry.hpp"
#include "sam3.h"

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
    return SAM3_DEVICE_AUTO;  // "auto" / NULL / anything else
}

// Shared engine log level for verbose progress (upstream SAM3_LOG_LEVEL=2).
// Kept at INFO unless a caller explicitly raises it via aicore_set_log_level.
constexpr int kSam3LogLevel = 1;

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

    // Cached encoded image (image mode). Re-encoded when the input size
    // changes so repeated prompts on one image reuse the backbone.
    bool encoded = false;
    int32_t encoded_w = 0;
    int32_t encoded_h = 0;
    std::vector<uint8_t> encoded_rgb;
};

AICORE_CAPI aicore_sam3_ctx* aicore_sam3_load_opts(
        const char* gguf_path, const aicore_sam3_options* opts) {
    if (!gguf_path || !gguf_path[0]) {
        AICORE_LOG_ERROR("[sam3] ", "load: empty model path\n");
        return nullptr;
    }

    std::unique_ptr<aicore_sam3_ctx> ctx(new (std::nothrow) aicore_sam3_ctx());
    if (!ctx) {
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
        ctx->last_error = "sam3_load_model failed (see console for details)";
        AICORE_LOG_ERROR("[sam3] ", "%s\n", ctx->last_error.c_str());
        return nullptr;
    }

    ctx->state = sam3_create_state(*ctx->model, params);
    if (!ctx->state) {
        ctx->last_error = "sam3_create_state failed";
        AICORE_LOG_ERROR("[sam3] ", "%s\n", ctx->last_error.c_str());
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

// Ensure the ctx has an encoded image for (width, height); re-encodes when
// the size changed or nothing is cached. pvs_only controls the neck.
static bool ensure_encoded(aicore_sam3_ctx* ctx,
                           const uint8_t* rgb,
                           int32_t width,
                           int32_t height,
                           size_t row_stride_bytes,
                           bool pvs_only) {
    if (ctx->encoded && ctx->encoded_w == width && ctx->encoded_h == height) {
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
    (void)what;
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
        ctx->last_error = "PCS requires a text prompt or at least one exemplar box";
        return nullptr;
    }
    // Mirror upstream examples/main_image.cpp: the text prompt is optional
    // when exemplar boxes are provided (and vice versa). Only reject when
    // both are empty — sam3_segment_pcs handles a missing text prompt by
    // falling back to SOT/EOT-only tokens.
    const bool no_text =
            !prompt->text || !prompt->text[0];
    const bool no_exemplars = prompt->n_pos_exemplars <= 0 &&
                              prompt->n_neg_exemplars <= 0;
    if (no_text && no_exemplars) {
        ctx->last_error = "PCS requires a text prompt or at least one exemplar box";
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
        return nullptr;
    }
    t->state = sam3_create_state(*t->model, t->params);
    if (!t->state) {
        t->last_error = "sam3_create_state failed";
        return nullptr;
    }
    return t.release();
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
    sam3_result r =
            sam3_segment_pvs(*tracker->state, *tracker->model, p);
    auto t1 = Clock::now();
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
        {"sam3-f16.gguf", "sam3", 1837900000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam3-q8_0.gguf", "sam3", 1099400000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam3-q4_1.gguf", "sam3", 755800000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam3-q4_0.gguf", "sam3", 706700000LL,
         "Q4_0 \xe2\x80\x94 smallest SAM3 quant"},
        // SAM 3 visual-only (no text encoder)
        {"sam3-visual-f16.gguf", "sam3-visual", 945500000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam3-visual-q8_0.gguf", "sam3-visual", 517100000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam3-visual-q4_1.gguf", "sam3-visual", 317700000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam3-visual-q4_0.gguf", "sam3-visual", 289200000LL,
         "Q4_0 \xe2\x80\x94 smallest SAM3-visual quant"},
        // SAM 2.1 (Hiera backbone, visual-only)
        {"sam2.1_hiera_large_f16.gguf", "sam2.1", 450900000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_large_f32.gguf", "sam2.1", 897800000LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_large_q8_0.gguf", "sam2.1", 241200000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_large_q4_1.gguf", "sam2.1", 143900000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_large_q4_0.gguf", "sam2.1", 130000000LL,
         "Q4_0 \xe2\x80\x94 smallest large quant"},
        {"sam2.1_hiera_base_plus_f16.gguf", "sam2.1", 163300000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_base_plus_f32.gguf", "sam2.1", 323400000LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_base_plus_q8_0.gguf", "sam2.1", 87700000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_base_plus_q4_1.gguf", "sam2.1", 53000000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_base_plus_q4_0.gguf", "sam2.1", 48000000LL,
         "Q4_0 \xe2\x80\x94 smallest base-plus quant"},
        {"sam2.1_hiera_small_f16.gguf", "sam2.1", 93600000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_small_f32.gguf", "sam2.1", 184300000LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_small_q8_0.gguf", "sam2.1", 50200000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_small_q4_1.gguf", "sam2.1", 30500000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_small_q4_0.gguf", "sam2.1", 27700000LL,
         "Q4_0 \xe2\x80\x94 smallest small quant"},
        {"sam2.1_hiera_tiny_f16.gguf", "sam2.1", 79300000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2.1_hiera_tiny_f32.gguf", "sam2.1", 155900000LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2.1_hiera_tiny_q8_0.gguf", "sam2.1", 42600000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2.1_hiera_tiny_q4_1.gguf", "sam2.1", 26000000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2.1_hiera_tiny_q4_0.gguf", "sam2.1", 23600000LL,
         "Q4_0 \xe2\x80\x94 smallest SAM2.1 quant (22 MB)"},
        // SAM 2 (Hiera backbone, visual-only)
        {"sam2_hiera_large_f16.gguf", "sam2", 450900000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2_hiera_base_plus_f16.gguf", "sam2", 163200000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2_hiera_base_plus_f32.gguf", "sam2", 323400000LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2_hiera_base_plus_q8_0.gguf", "sam2", 87700000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2_hiera_base_plus_q4_1.gguf", "sam2", 53000000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2_hiera_base_plus_q4_0.gguf", "sam2", 48000000LL,
         "Q4_0 \xe2\x80\x94 smallest base-plus quant"},
        {"sam2_hiera_tiny_f16.gguf", "sam2", 79300000LL,
         "F16 \xe2\x80\x94 half precision (recommended)"},
        {"sam2_hiera_tiny_f32.gguf", "sam2", 155800000LL,
         "F32 \xe2\x80\x94 full precision reference"},
        {"sam2_hiera_tiny_q8_0.gguf", "sam2", 42600000LL,
         "Q8_0 \xe2\x80\x94 8-bit quant, best accuracy/size trade"},
        {"sam2_hiera_tiny_q4_1.gguf", "sam2", 26000000LL,
         "Q4_1 \xe2\x80\x94 4-bit quant with bias"},
        {"sam2_hiera_tiny_q4_0.gguf", "sam2", 23600000LL,
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

}  // namespace

AICORE_CAPI int aicore_sam3_model_count(void) { return kModelCount; }

AICORE_CAPI const aicore_sam3_model_entry* aicore_sam3_model_at(int index) {
    if (index < 0 || index >= kModelCount) {
        return nullptr;
    }
    static std::vector<aicore_sam3_model_entry> entries;
    static bool initialized = false;
    if (!initialized) {
        entries.reserve(kModelCount);
        for (int i = 0; i < kModelCount; ++i) {
            const ModelEntry& m = kModels[i];
            static std::vector<std::string> urls;
            urls.emplace_back(std::string(kSamDownloadBase) + m.filename);
            aicore_sam3_model_entry e{};
            e.filename = m.filename;
            e.download_url = urls.back().c_str();
            e.display_name = familyDisplayName(m.family);
            e.quant_note = m.quant_note;
            e.model_family = m.family;
            e.size_bytes = m.size_bytes;
            e.visual_only = std::strcmp(m.family, "sam3") != 0 ? 1 : 0;
            entries.push_back(e);
        }
        initialized = true;
    }
    return &entries[index];
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

AICORE_CAPI int aicore_sam3_warmup_backend(const char* device) {
    // Probe the requested device and clear any stale CUDA error state, so
    // plugin startup can validate the backend before the first load.
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_sam3_shutdown(void) {
    // The engine keeps no process-wide backend leases beyond ggml's registry;
    // purging them is handled by the shared runtime.
    aicore::runtime::purge_inactive_backend_leases();
}
