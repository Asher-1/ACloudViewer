// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#include <QImage>
#include <QString>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/yolo_capi.h"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_registry.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "tasks/yolo/backend.hpp"
#include "tasks/yolo/yolo_graph.hpp"
#include "tasks/yolo/yolo_image.hpp"
#include "tasks/yolo/yolo_postprocess.hpp"

struct aicore_yolo_ctx {
    yolo::Session* engine = nullptr;
    std::string model_path;
    std::string device;
    int32_t threads = 0;
    std::string last_error;

    // Detection thresholds: the single configuration point for detect and
    // segment calls (defaults 0.25 / 0.7 / model max_det). Seeded from the
    // options struct at load; adjustable at runtime via
    // aicore_yolo_set_detect_thresholds without rebuilding the context.
    float conf_thres = 0.25f;
    float iou_thres = 0.7f;
    uint32_t top_k = 0;

    // Per-stage wall-clock timings of the most recent inference call
    // (mirrors the upstream ultralytics-ggml bench fields; surfaced through
    // aicore_yolo_last_timings for 1:1 latency comparisons).
    aicore_yolo_timings timings{};
    bool has_timings = false;

    // Statistics of the most recent depth call (surfaced through
    // aicore_yolo_last_depth_json; the float map itself is handed to the
    // caller and not kept here).
    struct DepthStats {
        bool valid = false;
        int width = 0, height = 0;
        int image_w = 0, image_h = 0;
        float min_d = 0, max_d = 0, mean_d = 0, p95_d = 0;
        size_t valid_pixels = 0;
    };
    DepthStats depth;
};

struct aicore_yolo_options {
    aicore::capi::CommonOptions common;
    float conf_thres = 0.25f;
    float iou_thres = 0.7f;
    uint32_t top_k = 0;
    // Mirrors the yolo::SessionOptions debug/tuning fields (complete bridge;
    // defaults equal the SessionOptions defaults).
    int log_level = 1;  // 0=DEBUG,1=INFO,2=WARN,3=ERROR
    int input_w = 0;    // 0: square imgsz from GGUF metadata
    int input_h = 0;
    bool keep_all_ops = false;
    bool profile_ops = false;
    bool profile_gaps = false;
};

struct aicore_yolo_segment_result {
    std::vector<yolo::Detection> dets;
    std::vector<yolo::SegMask> masks;
    // Canvas-space mask data (absolute coordinates), also stored per-mask
    int canvas_w = 0;
    int canvas_h = 0;
    // Class-name table copied from the session's model metadata, so
    // aicore_yolo_seg_det_class_name stays valid for the result's lifetime
    // (the typed API has no ctx handle to query after the call).
    std::vector<std::string> class_names;
};

using aicore::capi::dup_cstr;
using aicore::capi::json_escape;

namespace {

// Load an image file and hand the tightly-packed RGB buffer to f(rgb, w, h).
// The buffer is freed before returning; f must not keep a pointer to it.
template <class F>
auto with_path_rgb(const char* image_path,
                   aicore_yolo_ctx* ctx,
                   F&& f) -> decltype(f(nullptr, 0, 0)) {
    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) {
        ctx->last_error = std::string("failed to load image: ") + image_path;
        return decltype(f(nullptr, 0, 0))();
    }
    aicore::capi::PackedRgb packed = aicore::capi::qimage_to_packed_rgb(img);
    if (packed.data == nullptr) {
        ctx->last_error = "out of memory decoding image";
        return decltype(f(nullptr, 0, 0))();
    }
    auto result = f(packed.data, packed.width, packed.height);
    std::free(packed.data);
    return result;
}

}  // namespace

AICORE_CAPI int aicore_yolo_abi_version(void) { return 2; }

AICORE_CAPI aicore_yolo_options* aicore_yolo_options_new(void) {
    return new (std::nothrow) aicore_yolo_options();
}

AICORE_CAPI void aicore_yolo_options_free(aicore_yolo_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_yolo_options_set_device(aicore_yolo_options* opts,
                                                const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_yolo_options_set_threads(aicore_yolo_options* opts,
                                                 int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI void aicore_yolo_options_set_conf_thres(aicore_yolo_options* opts,
                                                    float conf_thres) {
    if (opts != nullptr && conf_thres > 0.0f && conf_thres < 1.0f)
        opts->conf_thres = conf_thres;
}

AICORE_CAPI void aicore_yolo_options_set_iou_thres(aicore_yolo_options* opts,
                                                   float iou_thres) {
    if (opts != nullptr && iou_thres > 0.0f && iou_thres < 1.0f)
        opts->iou_thres = iou_thres;
}

AICORE_CAPI void aicore_yolo_options_set_top_k(aicore_yolo_options* opts,
                                               uint32_t top_k) {
    if (opts != nullptr) opts->top_k = top_k;
}

AICORE_CAPI void aicore_yolo_options_set_log_level(aicore_yolo_options* opts,
                                                   int log_level) {
    if (opts == nullptr) return;
    if (log_level < 0 || log_level > 3) return;  // invalid keeps current
    opts->log_level = log_level;
}

AICORE_CAPI void aicore_yolo_options_set_input_size(aicore_yolo_options* opts,
                                                    int width,
                                                    int height) {
    if (opts == nullptr) return;
    if (width <= 0 || height <= 0) {  // 0/invalid clears to model default
        opts->input_w = 0;
        opts->input_h = 0;
        return;
    }
    opts->input_w = width;
    opts->input_h = height;
}

AICORE_CAPI void aicore_yolo_options_set_keep_all_ops(aicore_yolo_options* opts,
                                                      int enabled) {
    if (opts != nullptr && enabled >= 0) opts->keep_all_ops = enabled != 0;
}

AICORE_CAPI void aicore_yolo_options_set_profile_ops(aicore_yolo_options* opts,
                                                     int enabled) {
    if (opts != nullptr && enabled >= 0) opts->profile_ops = enabled != 0;
}

AICORE_CAPI void aicore_yolo_options_set_profile_gaps(aicore_yolo_options* opts,
                                                      int enabled) {
    if (opts != nullptr && enabled >= 0) opts->profile_gaps = enabled != 0;
}

AICORE_CAPI float aicore_yolo_options_get_conf_thres(
        const aicore_yolo_options* opts) {
    return opts != nullptr ? opts->conf_thres : 0.25f;
}

AICORE_CAPI float aicore_yolo_options_get_iou_thres(
        const aicore_yolo_options* opts) {
    return opts != nullptr ? opts->iou_thres : 0.7f;
}

AICORE_CAPI aicore_yolo_ctx* aicore_yolo_load_opts(
        const char* gguf_path, const aicore_yolo_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_yolo_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device = opts != nullptr ? opts->common.device : "auto";
    ctx->threads = opts != nullptr ? opts->common.threads : 0;
    if (opts != nullptr) {
        ctx->conf_thres = opts->conf_thres;
        ctx->iou_thres = opts->iou_thres;
        ctx->top_k = opts->top_k;
    }

    try {
        yolo::SessionOptions sopts;
        if (opts != nullptr) {
            sopts.threads = opts->common.threads;
            sopts.input_w = opts->input_w;
            sopts.input_h = opts->input_h;
            sopts.log_level = opts->log_level;
            sopts.keep_all_ops = opts->keep_all_ops;
            sopts.profile_ops = opts->profile_ops;
            sopts.profile_gaps = opts->profile_gaps;
        } else {
            sopts.threads = ctx->threads;
        }
        ctx->engine = yolo::create_session(gguf_path, ctx->device, sopts);
        if (ctx->engine == nullptr) {
            ctx->last_error = "failed to load YOLO GGUF";
        }
    } catch (const std::exception& e) {
        ctx->last_error = e.what();
    }
    return ctx;
}

AICORE_CAPI void aicore_yolo_free(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr) return;
    if (ctx->engine != nullptr) yolo::free_session(ctx->engine);
    delete ctx;
}

AICORE_CAPI int aicore_yolo_is_ready(const aicore_yolo_ctx* ctx) {
    return ctx != nullptr && ctx->engine != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_yolo_last_error(const aicore_yolo_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_yolo_free_buffer(void* p) { std::free(p); }

AICORE_CAPI int aicore_yolo_load_path_rgb(const char* image_path,
                                          uint8_t** out_rgb,
                                          int32_t* out_width,
                                          int32_t* out_height) {
    if (image_path == nullptr || out_rgb == nullptr || out_width == nullptr ||
        out_height == nullptr) {
        return -1;
    }
    *out_rgb = nullptr;
    *out_width = 0;
    *out_height = 0;
    QImage img(QString::fromUtf8(image_path));
    if (img.isNull()) return -1;
    aicore::capi::PackedRgb packed = aicore::capi::qimage_to_packed_rgb(img);
    if (packed.data == nullptr) return -1;
    *out_rgb = packed.data;
    *out_width = packed.width;
    *out_height = packed.height;
    return 0;
}

namespace {

// Shared detect core: letterbox, inference, postprocess, JSON envelope.
// Everything may allocate; an uncaught bad_alloc would cross the extern "C"
// boundary and the Qt event loop (queued worker slot) and terminate the
// process with SIGABRT — hence the catch fencing.
char* run_detect(aicore_yolo_ctx* ctx,
                 const uint8_t* rgb,
                 int32_t width,
                 int32_t height,
                 int* out_rc) {
    *out_rc = -1;
    yolo::Session* s = ctx->engine;
    if (s == nullptr || rgb == nullptr || width <= 0 || height <= 0) {
        return nullptr;
    }
    if (s->model.meta.task != "detect") {
        ctx->last_error =
                "model is not a detect model (task=" + s->model.meta.task + ")";
        return nullptr;
    }
    ctx->depth.valid = false;

    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(yolo::Image{width, height, rgb},
                              s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        if (!yolo::session_run(s, canvas.data())) {
            ctx->last_error = "YOLO inference failed";
            return nullptr;
        }
        std::vector<float> raw;
        int no = 0, na = 0;
        if (!yolo::session_read_output(s, raw, no, na)) {
            ctx->last_error = "output readback failed";
            return nullptr;
        }
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        yolo::PostprocConfig cfg;
        cfg.conf_thres = ctx->conf_thres;
        cfg.iou_thres = ctx->iou_thres;
        cfg.max_det = ctx->top_k > 0 ? (int)ctx->top_k : s->model.meta.max_det;
        std::vector<yolo::Detection> dets =
                yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(),
                                  s->anchor_strides.data(), cfg);
        yolo::unscale_boxes(dets, info);
        const double postprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        const auto& names = s->model.meta.class_names;
        std::ostringstream o;
        o << "{\"model\":\"" << json_escape(s->model.meta.name) << "\","
          << "\"task\":\"detect\","
          << "\"image_size\":" << s->model.meta.imgsz << ","
          << "\"num_classes\":" << s->model.meta.nc << ","
          << "\"end2end\":" << (s->model.meta.end2end ? 1 : 0) << ","
          << "\"image\":{\"width\":" << width << ",\"height\":" << height
          << "},\"detections\":[";
        for (size_t i = 0; i < dets.size(); ++i) {
            const auto& d = dets[i];
            std::string cname =
                    d.class_id >= 0 && d.class_id < (int)names.size()
                            ? names[d.class_id]
                            : ("class " + std::to_string(d.class_id));
            if (i) o << ',';
            o << "{\"class_id\":" << d.class_id << ",\"class_name\":\""
              << json_escape(cname) << "\",\"score\":" << d.score
              << ",\"box\":[" << d.x1 << ',' << d.y1 << ',' << d.x2 << ','
              << d.y2 << "]}";
        }
        o << "]}";
        char* json = dup_cstr(o.str());
        const double json_ms = yolo::ms_since(t0);
        const double e2e_ms = yolo::ms_since(t_e2e);
        ctx->timings = aicore_yolo_timings{preprocess_ms, inference_ms,
                                           postprocess_ms, json_ms, e2e_ms};
        ctx->has_timings = true;
        *out_rc = 0;
        return json;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error =
                std::string("YOLO post-processing error: ") + e.what();
        return nullptr;
    }
}

// Shared depth core: letterbox, inference, model-resolution readback, restore
// to the original image size, summary statistics, malloc'd float array.
float* run_depth(aicore_yolo_ctx* ctx,
                 const uint8_t* rgb,
                 int32_t width,
                 int32_t height,
                 int32_t* out_width,
                 int32_t* out_height) {
    yolo::Session* s = ctx->engine;
    if (s == nullptr || rgb == nullptr || width <= 0 || height <= 0) {
        return nullptr;
    }
    if (s->model.meta.task != "depth") {
        ctx->last_error =
                "model is not a depth model (task=" + s->model.meta.task + ")";
        return nullptr;
    }

    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(yolo::Image{width, height, rgb},
                              s->model.meta.imgsz, info, canvas);
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild for the letterbox canvas failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        if (!yolo::session_run(s, canvas.data())) {
            ctx->last_error = "YOLO inference failed";
            return nullptr;
        }
        std::vector<float> model_depth;
        int dw = 0, dh = 0;
        if (!yolo::session_read_depth(s, model_depth, dw, dh)) {
            ctx->last_error = "depth readback failed";
            return nullptr;
        }
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        std::vector<float> restored =
                yolo::restore_depth(model_depth, dw, dh, info, width, height);
        if (restored.empty()) {
            ctx->last_error = "depth restoration failed";
            return nullptr;
        }

        // Summary statistics over valid pixels (finite && > 0), mirroring the
        // upstream write_depth_png normalization inputs.
        std::vector<float> valid;
        valid.reserve(restored.size());
        double sum = 0.0;
        for (float v : restored) {
            if (std::isfinite(v) && v > 0.0f) {
                valid.push_back(v);
                sum += v;
            }
        }
        ctx->depth.valid = true;
        ctx->depth.width = width;
        ctx->depth.height = height;
        ctx->depth.image_w = width;
        ctx->depth.image_h = height;
        ctx->depth.valid_pixels = valid.size();
        if (!valid.empty()) {
            const float mn = *std::min_element(valid.begin(), valid.end());
            const float mx = *std::max_element(valid.begin(), valid.end());
            const size_t p95 =
                    std::min(valid.size() - 1, valid.size() * 95 / 100);
            std::nth_element(valid.begin(), valid.begin() + p95, valid.end());
            ctx->depth.min_d = mn;
            ctx->depth.max_d = mx;
            ctx->depth.mean_d = (float)(sum / (double)valid.size());
            ctx->depth.p95_d = valid[p95];
        } else {
            ctx->depth.min_d = ctx->depth.max_d = ctx->depth.mean_d =
                    ctx->depth.p95_d = 0.0f;
        }

        float* out = static_cast<float*>(
                std::malloc(restored.size() * sizeof(float)));
        if (out == nullptr) {
            ctx->depth.valid = false;
            ctx->last_error = "YOLO out of memory returning depth map";
            return nullptr;
        }
        std::memcpy(out, restored.data(), restored.size() * sizeof(float));
        if (out_width != nullptr) *out_width = width;
        if (out_height != nullptr) *out_height = height;
        const double postprocess_ms = yolo::ms_since(t0);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return out;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in depth post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO depth error: ") + e.what();
        return nullptr;
    }
}

}  // namespace

AICORE_CAPI char* aicore_yolo_detect_path_json(aicore_yolo_ctx* ctx,
                                               const char* image_path) {
    if (ctx == nullptr || ctx->engine == nullptr || image_path == nullptr) {
        return nullptr;
    }
    return with_path_rgb(image_path, ctx,
                         [&](const uint8_t* rgb, int w, int h) {
                             int rc = -1;
                             return run_detect(ctx, rgb, w, h, &rc);
                         });
}

AICORE_CAPI char* aicore_yolo_detect_rgb_json(aicore_yolo_ctx* ctx,
                                              const uint8_t* rgb,
                                              int32_t width,
                                              int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    int rc = -1;
    /* Borrow (no copy): the caller's buffer must stay alive for the whole
     * call, which the synchronous C API contract guarantees (preprocess
     * only reads from it). Saves a full-frame copy per detection call. */
    return run_detect(ctx, rgb, width, height, &rc);
}

/** Runtime threshold update without rebuilding the context (validated: out
 *  of range values keep the previous value). */
AICORE_CAPI void aicore_yolo_set_detect_thresholds(aicore_yolo_ctx* ctx,
                                                   float conf_thres,
                                                   float iou_thres,
                                                   uint32_t top_k) {
    if (ctx == nullptr) return;
    if (conf_thres > 0.0f && conf_thres < 1.0f) ctx->conf_thres = conf_thres;
    if (iou_thres > 0.0f && iou_thres < 1.0f) ctx->iou_thres = iou_thres;
    ctx->top_k = top_k;
}

/** Drop the host-side copies of the model weights (halves the host memory
 *  footprint; the device weight buffer is untouched, so inference keeps
 *  working). Reload on demand with aicore_yolo_ensure_host_weights. */
AICORE_CAPI int aicore_yolo_release_host_weights(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return -1;
    return yolo::session_release_host_weights(ctx->engine) ? 0 : -1;
}

/** Reload released host weight copies from the GGUF file (no-op when they
 *  are present). Returns 0 on success, -1 when the GGUF cannot be reopened. */
AICORE_CAPI int aicore_yolo_ensure_host_weights(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return -1;
    return yolo::session_ensure_host_weights(ctx->engine) ? 0 : -1;
}

AICORE_CAPI float* aicore_yolo_depth_path(aicore_yolo_ctx* ctx,
                                          const char* image_path,
                                          int32_t* out_width,
                                          int32_t* out_height) {
    if (ctx == nullptr || ctx->engine == nullptr || image_path == nullptr) {
        return nullptr;
    }
    return with_path_rgb(
            image_path, ctx, [&](const uint8_t* rgb, int w, int h) {
                return run_depth(ctx, rgb, w, h, out_width, out_height);
            });
}

AICORE_CAPI float* aicore_yolo_depth_rgb(aicore_yolo_ctx* ctx,
                                         const uint8_t* rgb,
                                         int32_t width,
                                         int32_t height,
                                         int32_t* out_width,
                                         int32_t* out_height) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    if (out_width != nullptr) *out_width = 0;
    if (out_height != nullptr) *out_height = 0;
    return run_depth(ctx, rgb, width, height, out_width, out_height);
}

// ---- Segment result (typed API) ----

AICORE_CAPI aicore_yolo_segment_result* aicore_yolo_seg_rgb(
        aicore_yolo_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height) {
    if (ctx == nullptr || ctx->engine == nullptr || rgb == nullptr ||
        width <= 0 || height <= 0) {
        return nullptr;
    }
    yolo::Session* s = ctx->engine;
    if (s->model.meta.task != "segment") {
        ctx->last_error =
                "model is not a segment model (task=" + s->model.meta.task +
                ")";
        return nullptr;
    }

    try {
        const auto t_e2e = yolo::Clock::now();
        auto t0 = yolo::Clock::now();
        yolo::LetterboxInfo info;
        std::vector<float> canvas;
        yolo::letterbox_image(yolo::Image{width, height, rgb},
                              s->model.meta.imgsz, info, canvas);

        // Canvas resize if needed
        if (!yolo::session_ensure_canvas(s, info.imgsz_w, info.imgsz_h)) {
            ctx->last_error = "graph rebuild failed";
            return nullptr;
        }
        const double preprocess_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        if (!yolo::session_run(s, canvas.data())) {
            ctx->last_error = "YOLO segment inference failed";
            return nullptr;
        }

        // Read detect output
        std::vector<float> raw;
        int no = 0, na = 0;
        if (!yolo::session_read_output(s, raw, no, na)) {
            ctx->last_error = "output readback failed";
            return nullptr;
        }
        const double inference_ms = yolo::ms_since(t0);

        t0 = yolo::Clock::now();
        // Read proto output — counted as postprocess, mirroring the upstream
        // bench (proto readback + mask composition belong to post_ms).
        std::vector<float> proto;
        int nm = 0, proto_w = 0, proto_h = 0;
        if (!yolo::session_read_proto(s, proto, nm, proto_w, proto_h)) {
            ctx->last_error = "proto readback failed";
            return nullptr;
        }

        // Postprocess
        yolo::PostprocConfig cfg;
        cfg.conf_thres = ctx->conf_thres;
        cfg.iou_thres = ctx->iou_thres;
        cfg.max_det = ctx->top_k > 0 ? (int)ctx->top_k : s->model.meta.max_det;
        std::vector<yolo::Detection> dets =
                yolo::postprocess(raw, no, na, s->model.meta, s->anchors.data(),
                                  s->anchor_strides.data(), cfg);

        // Compose masks (before unscale_boxes — masks are in canvas coords)
        std::vector<yolo::SegMask> masks = yolo::compose_masks(
                dets, raw, na, s->model.meta, proto, proto_w, proto_h,
                info.imgsz_w, info.imgsz_h);

        // Unscale boxes to original image coordinates
        yolo::unscale_boxes(dets, info);

        // Masks follow the boxes into the original image space: the canvas
        // windows compose_masks produced would overlay the wrong region on
        // the source image (canvas dims differ from source dims, and the
        // window origins are lost across the C API). Full-size source masks
        // make the typed result directly drawable at 1:1 with the boxes.
        yolo::unscale_masks(masks, info, width, height);

        auto* res = new (std::nothrow) aicore_yolo_segment_result();
        if (!res) {
            ctx->last_error = "YOLO out of memory for segment result";
            return nullptr;
        }
        res->dets = std::move(dets);
        res->masks = std::move(masks);
        res->canvas_w = info.imgsz_w;
        res->canvas_h = info.imgsz_h;
        // Copy the model's class table so seg_det_class_name can serve
        // names for the whole result lifetime (no ctx dependency).
        res->class_names = s->model.meta.class_names;
        const double postprocess_ms = yolo::ms_since(t0);
        ctx->timings =
                aicore_yolo_timings{preprocess_ms, inference_ms, postprocess_ms,
                                    0.0, yolo::ms_since(t_e2e)};
        ctx->has_timings = true;
        return res;
    } catch (const std::bad_alloc&) {
        ctx->last_error = "YOLO out of memory in segment post-processing";
        return nullptr;
    } catch (const std::exception& e) {
        ctx->last_error = std::string("YOLO segment error: ") + e.what();
        return nullptr;
    }
}

AICORE_CAPI int aicore_yolo_seg_det_count(
        const aicore_yolo_segment_result* res) {
    return res != nullptr ? (int)res->dets.size() : 0;
}

AICORE_CAPI aicore_yolo_detection
aicore_yolo_seg_det_at(const aicore_yolo_segment_result* res, int index) {
    aicore_yolo_detection det = {};
    if (res != nullptr && index >= 0 && index < (int)res->dets.size()) {
        const auto& d = res->dets[index];
        det.x1 = d.x1;
        det.y1 = d.y1;
        det.x2 = d.x2;
        det.y2 = d.y2;
        det.score = d.score;
        det.class_id = d.class_id;
    }
    return det;
}

AICORE_CAPI const char* aicore_yolo_seg_det_class_name(
        const aicore_yolo_segment_result* res, int index) {
    if (res == nullptr || index < 0 || index >= (int)res->dets.size()) {
        return nullptr;
    }
    const int cid = res->dets[index].class_id;
    if (cid < 0 || cid >= (int)res->class_names.size()) {
        return nullptr;
    }
    const std::string& name = res->class_names[cid];
    return name.empty() ? nullptr : name.c_str();
}

AICORE_CAPI aicore_yolo_plane_view
aicore_yolo_seg_mask_at(const aicore_yolo_segment_result* res, int index) {
    aicore_yolo_plane_view view = {};
    if (res != nullptr && index >= 0 && index < (int)res->masks.size()) {
        const auto& m = res->masks[index];
        view.data = m.bits.data();
        view.width = m.w;
        view.height = m.h;
        view.row_stride_bytes = (size_t)m.w;
    }
    return view;
}

AICORE_CAPI void aicore_yolo_seg_result_free(aicore_yolo_segment_result* res) {
    delete res;
}

AICORE_CAPI char* aicore_yolo_last_depth_json(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr || !ctx->depth.valid) {
        return nullptr;
    }
    try {
        const auto& st = ctx->depth;
        std::ostringstream o;
        o << "{\"model\":\"" << json_escape(ctx->engine->model.meta.name)
          << "\","
          << "\"task\":\"depth\","
          << "\"image_size\":" << ctx->engine->model.meta.imgsz << ","
          << "\"image\":{\"width\":" << st.image_w
          << ",\"height\":" << st.image_h << "},"
          << "\"depth_width\":" << st.width << ",\"depth_height\":" << st.height
          << ","
          << "\"min_depth\":" << st.min_d << ",\"max_depth\":" << st.max_d
          << ","
          << "\"mean_depth\":" << st.mean_d << ",\"p95_depth\":" << st.p95_d
          << ","
          << "\"valid_pixels\":" << st.valid_pixels << "}";
        return dup_cstr(o.str());
    } catch (const std::exception& e) {
        ctx->last_error =
                std::string("depth stats serialization failed: ") + e.what();
        return nullptr;
    }
}

AICORE_CAPI const char* aicore_yolo_context_task(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? ctx->engine->model.meta.task.c_str()
                   : "";
}

AICORE_CAPI const char* aicore_yolo_context_model_name(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? ctx->engine->model.meta.name.c_str()
                   : "";
}

AICORE_CAPI uint32_t aicore_yolo_context_image_size(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? (uint32_t)ctx->engine->model.meta.imgsz
                   : 0;
}

AICORE_CAPI uint32_t aicore_yolo_context_num_classes(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr)
                   ? (uint32_t)ctx->engine->model.meta.nc
                   : 0;
}

AICORE_CAPI int aicore_yolo_context_end2end(aicore_yolo_ctx* ctx) {
    return (ctx != nullptr && ctx->engine != nullptr &&
            ctx->engine->model.meta.end2end)
                   ? 1
                   : 0;
}

AICORE_CAPI const char* aicore_yolo_context_device(aicore_yolo_ctx* ctx) {
    /* Pointer owned by ctx (stable until the next load); callers must copy
     * it before the context is freed. */
    if (ctx == nullptr || ctx->engine == nullptr) return "";
    const std::string& resolved = ctx->engine->backend.device_name;
    return resolved.empty() ? "" : resolved.c_str();
}

AICORE_CAPI int aicore_yolo_context_threads(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return 0;
    return ctx->engine->backend.n_threads;
}

AICORE_CAPI char* aicore_yolo_info_json(aicore_yolo_ctx* ctx) {
    if (ctx == nullptr || ctx->engine == nullptr) return nullptr;
    const yolo::ModelMeta& meta = ctx->engine->model.meta;
    std::ostringstream o;
    o << "{\"model\":\"" << json_escape(meta.name) << "\","
      << "\"task\":\"" << json_escape(meta.task) << "\","
      << "\"image_size\":" << meta.imgsz << ","
      << "\"num_classes\":" << meta.nc << ","
      << "\"end2end\":" << (meta.end2end ? 1 : 0) << ","
      << "\"reg_max\":" << meta.reg_max << ","
      << "\"dtype\":\"" << json_escape(meta.dtype)
      << "\","
      /* The backend-RESOLVED device, not the request: makes a silent CPU
       * fallback visible to callers comparing against GPU latency
       * expectations. */
      << "\"device\":\"" << json_escape(aicore_yolo_context_device(ctx))
      << "\","
      << "\"threads\":" << ctx->engine->backend.n_threads << "}";
    return dup_cstr(o.str());
}

AICORE_CAPI int aicore_yolo_last_timings(const aicore_yolo_ctx* ctx,
                                         aicore_yolo_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_timings) {
        return -1;
    }
    *out_timings = ctx->timings;
    return 0;
}

AICORE_CAPI int aicore_yolo_warmup_backend(const char* device) {
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI void aicore_yolo_shutdown(void) {
    // Reclaims process-wide backend registry entries whose owners are gone
    // (expired leases). Live contexts are never touched. There is no other
    // process-global YOLO state to release; ggml backends themselves are
    // registered for the process lifetime by ggml_backend_load_all.
    aicore::runtime::purge_inactive_backend_leases();
}

AICORE_CAPI char* aicore_yolo_model_cache_dir(void) {
    return dup_cstr(aicore::yolo_model_cache_dir());
}
