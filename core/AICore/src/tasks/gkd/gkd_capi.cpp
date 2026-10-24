// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// GKD C API implementation over the internal C++ session (tasks/gkd/
// gkd_graph.hpp). The typed hot path is aicore_gkd_detect_image; the JSON
// and packed-RGB entry points are thin compatibility wrappers around it.

#include "aicore/gkd_capi.h"

#include <QImage>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "tasks/gkd/gkd_graph.hpp"

namespace {

using aicore::capi::dup_cstr;
using aicore::capi::json_escape;

}  // namespace

// Options/context layouts matching the opaque typedefs in aicore/gkd_capi.h.
// Global scope (not anonymous) — the C header already declares the tags.
struct aicore_gkd_options {
    aicore::capi::CommonOptions common;
    int log_level = 1;
    std::string dump_dir;
};

// Context-owned snapshot of the most recent detect call.
namespace {

struct LastResult {
    bool valid = false;
    int n_keypoints = 0;
    std::vector<float> kps_norm;       // N x 2 (-1..1 ROI space)
    std::vector<float> kps_pixel;      // N x 2 (source-image pixels)
    std::vector<float> scores;         // N
    std::vector<std::string> prompts;  // text prompts when given
    float roi[4] = {0, 0, 0, 0};       // used ROI in image pixels
};

}  // namespace

struct aicore_gkd_ctx {
    std::unique_ptr<gkd::GkdSession> session;
    std::string model_path;
    std::string model_name = "GKDT-L";
    std::string device;  // requested
    std::string resolved_device;
    int32_t threads = 0;
    std::string last_error;

    LastResult last;
    /** Per-ROI snapshots of the most recent detect call (size 1 for the
     *  single-ROI entry points, N for the multi-ROI batch). */
    std::vector<LastResult> last_all;
    aicore_gkd_timings timings{};
    aicore_pipeline_timings pipeline{};
    bool has_timings = false;
    bool has_pipeline = false;
};

namespace {

gkd::LogLevel log_level_from_int(int level) {
    if (level <= 0) return gkd::LogLevel::Debug;
    if (level == 1) return gkd::LogLevel::Info;
    if (level == 2) return gkd::LogLevel::Warn;
    return gkd::LogLevel::Error;
}

// Validates the request against the contract and fills the engine input.
// Returns an error message (empty on success).
std::string validate_request(const aicore_gkd_detect_request* req) {
    if (req == nullptr) return "detect request is NULL";
    if (req->struct_size != sizeof(aicore_gkd_detect_request)) {
        return "detect request struct_size mismatch";
    }
    if (req->n_kps_texts < 0) return "negative n_kps_texts";
    if (req->n_kps_texts > 0 && req->kps_texts == nullptr) {
        return "kps_texts is NULL with a positive count";
    }
    for (int32_t i = 0; i < req->n_kps_texts; ++i) {
        if (req->kps_texts[i] == nullptr) return "NULL text prompt";
    }
    const bool has_support = req->support_image != nullptr ||
                             req->n_support_kps > 0 ||
                             req->support_kps_xy != nullptr;
    if (req->n_support_kps < 0) return "negative n_support_kps";
    if (req->support_image != nullptr) {
        if (req->n_support_kps <= 0 || req->support_kps_xy == nullptr) {
            return "support image requires support keypoints";
        }
    } else if (has_support) {
        return "support keypoints given without a support image";
    }
    if (req->n_kps_texts == 0 && req->n_support_kps == 0) {
        return "no prompts given: need text prompts and/or a support image "
               "with keypoints";
    }
    if (req->n_kps_texts > 0 && req->n_support_kps > 0 &&
        req->n_kps_texts != req->n_support_kps) {
        return "multimodal prompts require matching counts (the official "
               "fuse pairs text row i with visual row i)";
    }
    return std::string();
}

// Rebuilds a per-ROI result snapshot from the typed engine output.
LastResult make_last_result(aicore_gkd_ctx* ctx,
                            const gkd::DetectOutput& out,
                            const std::vector<std::string>& prompts,
                            const float* bbox_or_null,
                            int32_t image_w,
                            int32_t image_h) {
    LastResult last;
    last.valid = true;
    last.n_keypoints = out.n_prompts;
    last.kps_norm = out.kps_norm;
    last.scores = out.scores;
    last.prompts = prompts;
    last.kps_pixel.assign((size_t)out.n_prompts * 2, 0.0f);
    if (out.n_prompts > 0) {
        gkd::recover_kps(out.kps_norm.data(), out.n_prompts,
                         ctx->session ? ctx->session->params().img_size : 0,
                         out.trans, last.kps_pixel.data());
    }
    if (bbox_or_null != nullptr) {
        std::memcpy(last.roi, bbox_or_null, sizeof(last.roi));
    } else {
        last.roi[0] = 0.0f;
        last.roi[1] = 0.0f;
        last.roi[2] = (float)(image_w - 1);
        last.roi[3] = (float)(image_h - 1);
    }
    return last;
}

}  // namespace

AICORE_CAPI int aicore_gkd_abi_version(void) { return 1; }

AICORE_CAPI aicore_gkd_options* aicore_gkd_options_new(void) {
    return new (std::nothrow) aicore_gkd_options();
}

AICORE_CAPI void aicore_gkd_options_free(aicore_gkd_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_gkd_options_set_device(aicore_gkd_options* opts,
                                               const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_gkd_options_set_threads(aicore_gkd_options* opts,
                                                int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI void aicore_gkd_options_set_log_level(aicore_gkd_options* opts,
                                                  int log_level) {
    if (opts != nullptr) opts->log_level = log_level;
}

AICORE_CAPI void aicore_gkd_options_set_dump_dir(aicore_gkd_options* opts,
                                                 const char* dir) {
    if (opts != nullptr) opts->dump_dir = dir != nullptr ? dir : "";
}

AICORE_CAPI aicore_gkd_ctx* aicore_gkd_load_opts(
        const char* gguf_path, const aicore_gkd_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_gkd_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device = opts != nullptr ? opts->common.device : "auto";
    ctx->threads = opts != nullptr ? opts->common.threads : 0;
    gkd::set_log_level(
            log_level_from_int(opts != nullptr ? opts->log_level : 1));

    std::string device_request = ctx->device.empty() ? "auto" : ctx->device;
    // VRAM admission guard (see the yolo capi for the failure mode this
    // prevents: a fatal backend abort under memory pressure at inference
    // time turns into a clean, actionable load error here).
    {
        std::error_code size_ec;
        const auto model_bytes = std::filesystem::file_size(gguf_path, size_ec);
        if (!size_ec &&
            !ggml_common::gpu_admission_check(device_request,
                                              static_cast<size_t>(model_bytes),
                                              &ctx->last_error)) {
            return ctx;
        }
    }
    ctx->session =
            gkd::GkdSession::create(gguf_path, ctx->threads, device_request);
    if (!ctx->session) {
        ctx->last_error =
                "failed to load GKDT GGUF: " + std::string(gguf_path) +
                " (see the [GKD] log lines for the reason)";
        return ctx;
    }
    ctx->resolved_device = ctx->session->backend();
    ctx->threads = ctx->session->threads();
    if (opts != nullptr && !opts->dump_dir.empty()) {
        ctx->session->set_dump_dir(opts->dump_dir);
    }
    return ctx;
}

AICORE_CAPI void aicore_gkd_free(aicore_gkd_ctx* ctx) { delete ctx; }

AICORE_CAPI int aicore_gkd_is_ready(const aicore_gkd_ctx* ctx) {
    return ctx != nullptr && ctx->session != nullptr ? 1 : 0;
}

AICORE_CAPI const char* aicore_gkd_last_error(const aicore_gkd_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_gkd_free_buffer(void* p) { std::free(p); }

namespace {

// Shared implementation of both detect entry points. `image_rgb` is a
// tightly-packed HWC RGB buffer owned by the caller.
int detect_impl(aicore_gkd_ctx* ctx,
                const uint8_t* image_rgb,
                int32_t width,
                int32_t height,
                const aicore_gkd_detect_request* req,
                const float* bboxes_xyxy = nullptr,
                int32_t n_bboxes = 0) {
    ctx->last_error.clear();
    ctx->last.valid = false;

    if (ctx->session == nullptr) {
        ctx->last_error = "no loaded model";
        return -1;
    }
    if (image_rgb == nullptr || width <= 0 || height <= 0) {
        ctx->last_error = "invalid image buffer";
        return -1;
    }
    std::string err = validate_request(req);
    if (!err.empty()) {
        ctx->last_error = err;
        return -1;
    }

    // Borrowed pixels -> tightly-packed RGB staging (single preprocess copy).
    aicore_image_view view{};
    view.data = image_rgb;
    view.width = width;
    view.height = height;
    view.row_stride_bytes = (size_t)width * 3;
    view.format = AICORE_IMAGE_RGB8;

    gkd::RgbImage query;
    if (!gkd::rgb_from_view(view, query)) {
        ctx->last_error = "invalid image buffer (preprocess staging failed)";
        return -1;
    }

    gkd::DetectInput extra;
    std::vector<std::string> prompts;
    prompts.reserve((size_t)req->n_kps_texts);
    for (int32_t i = 0; i < req->n_kps_texts; ++i) {
        extra.kps_texts.emplace_back(req->kps_texts[i]);
        prompts.emplace_back(req->kps_texts[i]);
    }
    if (req->support_image != nullptr) {
        if (!gkd::rgb_from_view(*req->support_image, extra.support_image)) {
            ctx->last_error = "invalid support image view";
            return -1;
        }
        extra.has_support = true;
        extra.support_kps_xy.assign(
                req->support_kps_xy,
                req->support_kps_xy + (size_t)req->n_support_kps * 2);
        extra.support_kps_vis.assign(req->n_support_kps, 1);
        if (req->support_kps_vis != nullptr) {
            std::memcpy(extra.support_kps_vis.data(), req->support_kps_vis,
                        (size_t)req->n_support_kps);
        }
    }

    std::vector<gkd::DetectOutput> outputs;
    gkd::StageTiming stage{};
    // ROI list: the multi entry passes its own array; the single entry
    // forwards the request's optional single box (n_bbox = has_box).
    std::vector<float> boxes;
    if (bboxes_xyxy != nullptr && n_bboxes > 0) {
        boxes.assign(bboxes_xyxy, bboxes_xyxy + (size_t)n_bboxes * 4);
    } else if (req->bbox_xyxy != nullptr) {
        boxes.assign(req->bbox_xyxy, req->bbox_xyxy + 4);
    }
    if (!ctx->session->detect(query, boxes.empty() ? nullptr : boxes.data(),
                              (int32_t)(boxes.size() / 4), extra, outputs,
                              &stage)) {
        ctx->last_error = "GKD inference failed (see the [GKD] log lines)";
        return -2;
    }
    if (outputs.empty()) {
        ctx->last_error = "GKD inference returned no result";
        return -2;
    }

    ctx->last_all.clear();
    ctx->last_all.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        const float* roi = boxes.empty() ? nullptr : boxes.data() + i * 4;
        ctx->last_all.push_back(
                make_last_result(ctx, outputs[i], prompts, roi, width, height));
    }
    // Legacy single-ROI accessors keep reading the first result.
    ctx->last = ctx->last_all.front();

    ctx->timings.preprocess_ms = stage.preprocess;
    ctx->timings.vision_ms = stage.vision;
    ctx->timings.text_ms = stage.text;
    ctx->timings.prompt_prep_ms = stage.prompt_prep;
    ctx->timings.detect_ms = stage.detect;
    ctx->timings.decode_ms = stage.decode;
    ctx->timings.e2e_ms = stage.total;
    ctx->has_timings = true;

    ctx->pipeline.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    ctx->pipeline.valid_fields = AICORE_TIMING_PREPROCESS |
                                 AICORE_TIMING_INFERENCE |
                                 AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    ctx->pipeline.preprocess_ms = stage.preprocess + stage.prompt_prep;
    ctx->pipeline.inference_ms = stage.vision + stage.text + stage.detect;
    ctx->pipeline.postprocess_ms = stage.decode;
    ctx->pipeline.serialization_ms = 0.0;
    ctx->pipeline.e2e_ms = stage.total;
    ctx->has_pipeline = true;
    return 0;
}

}  // namespace

AICORE_CAPI int aicore_gkd_detect_image(aicore_gkd_ctx* ctx,
                                        const aicore_image_view* image,
                                        const aicore_gkd_detect_request* req) {
    if (ctx == nullptr || image == nullptr) return -1;
    if (image->data == nullptr || image->width <= 0 || image->height <= 0) {
        ctx->last_error = "invalid image view";
        return -1;
    }
    gkd::RgbImage rgb;
    if (!gkd::rgb_from_view(*image, rgb)) {
        ctx->last_error = "unsupported image view (preprocess staging failed)";
        return -1;
    }
    return detect_impl(ctx, rgb.data.data(), rgb.w, rgb.h, req);
}

AICORE_CAPI int aicore_gkd_detect_rgb(aicore_gkd_ctx* ctx,
                                      const uint8_t* rgb,
                                      int32_t width,
                                      int32_t height,
                                      const aicore_gkd_detect_request* req) {
    if (ctx == nullptr) return -1;
    return detect_impl(ctx, rgb, width, height, req);
}

AICORE_CAPI int aicore_gkd_detect_image_multi(
        aicore_gkd_ctx* ctx,
        const aicore_image_view* image,
        const aicore_gkd_detect_request* base_req,
        const float* bboxes_xyxy,
        int32_t n_bboxes) {
    if (ctx == nullptr || image == nullptr) return -1;
    if (image->data == nullptr || image->width <= 0 || image->height <= 0) {
        ctx->last_error = "invalid image view";
        return -1;
    }
    if (bboxes_xyxy == nullptr || n_bboxes <= 0) {
        ctx->last_error = "multi-ROI detect requires at least one bbox";
        return -1;
    }
    gkd::RgbImage rgb;
    if (!gkd::rgb_from_view(*image, rgb)) {
        ctx->last_error = "unsupported image view (preprocess staging failed)";
        return -1;
    }
    return detect_impl(ctx, rgb.data.data(), rgb.w, rgb.h, base_req,
                       bboxes_xyxy, n_bboxes);
}

AICORE_CAPI int aicore_gkd_detect_rgb_multi(
        aicore_gkd_ctx* ctx,
        const uint8_t* rgb,
        int32_t width,
        int32_t height,
        const aicore_gkd_detect_request* base_req,
        const float* bboxes_xyxy,
        int32_t n_bboxes) {
    if (ctx == nullptr) return -1;
    if (bboxes_xyxy == nullptr || n_bboxes <= 0) {
        ctx->last_error = "multi-ROI detect requires at least one bbox";
        return -1;
    }
    return detect_impl(ctx, rgb, width, height, base_req, bboxes_xyxy,
                       n_bboxes);
}

AICORE_CAPI int32_t aicore_gkd_result_roi_count(const aicore_gkd_ctx* ctx) {
    return ctx != nullptr && !ctx->last_all.empty()
                   ? (int32_t)ctx->last_all.size()
                   : 0;
}

AICORE_CAPI int aicore_gkd_result_keypoint_count_at(const aicore_gkd_ctx* ctx,
                                                    int32_t roi) {
    if (ctx == nullptr || roi < 0 || (size_t)roi >= ctx->last_all.size()) {
        return 0;
    }
    const LastResult& r = ctx->last_all[(size_t)roi];
    return r.valid ? r.n_keypoints : 0;
}

AICORE_CAPI aicore_gkd_keypoint aicore_gkd_result_keypoint_at_roi(
        const aicore_gkd_ctx* ctx, int32_t roi, int index) {
    aicore_gkd_keypoint kp{};
    if (ctx == nullptr || roi < 0 || (size_t)roi >= ctx->last_all.size()) {
        return kp;
    }
    const LastResult& r = ctx->last_all[(size_t)roi];
    if (!r.valid || index < 0 || index >= r.n_keypoints) return kp;
    kp.x_norm = r.kps_norm[(size_t)index * 2 + 0];
    kp.y_norm = r.kps_norm[(size_t)index * 2 + 1];
    kp.x = r.kps_pixel[(size_t)index * 2 + 0];
    kp.y = r.kps_pixel[(size_t)index * 2 + 1];
    kp.score = r.scores[index];
    return kp;
}

AICORE_CAPI const char* aicore_gkd_result_prompt_at_roi(
        const aicore_gkd_ctx* ctx, int32_t roi, int index) {
    if (ctx == nullptr || roi < 0 || (size_t)roi >= ctx->last_all.size()) {
        return nullptr;
    }
    const LastResult& r = ctx->last_all[(size_t)roi];
    if (!r.valid || index < 0 || index >= (int)r.prompts.size()) {
        return nullptr;
    }
    return r.prompts[index].c_str();
}

AICORE_CAPI int aicore_gkd_result_roi_bbox_at(const aicore_gkd_ctx* ctx,
                                              int32_t roi,
                                              float out_bbox[4]) {
    if (ctx == nullptr || out_bbox == nullptr) return -1;
    std::memset(out_bbox, 0, sizeof(float) * 4);
    if (roi < 0 || (size_t)roi >= ctx->last_all.size()) return -1;
    const LastResult& r = ctx->last_all[(size_t)roi];
    if (!r.valid) return -1;
    std::memcpy(out_bbox, r.roi, sizeof(r.roi));
    return 0;
}

AICORE_CAPI int aicore_gkd_result_keypoint_count(const aicore_gkd_ctx* ctx) {
    return ctx != nullptr && ctx->last.valid ? ctx->last.n_keypoints : 0;
}

AICORE_CAPI aicore_gkd_keypoint
aicore_gkd_result_keypoint_at(const aicore_gkd_ctx* ctx, int index) {
    aicore_gkd_keypoint kp{};
    if (ctx == nullptr || !ctx->last.valid || index < 0 ||
        index >= ctx->last.n_keypoints) {
        return kp;
    }
    kp.x_norm = ctx->last.kps_norm[(size_t)index * 2 + 0];
    kp.y_norm = ctx->last.kps_norm[(size_t)index * 2 + 1];
    kp.x = ctx->last.kps_pixel[(size_t)index * 2 + 0];
    kp.y = ctx->last.kps_pixel[(size_t)index * 2 + 1];
    kp.score = ctx->last.scores[index];
    return kp;
}

AICORE_CAPI const char* aicore_gkd_result_prompt_at(const aicore_gkd_ctx* ctx,
                                                    int index) {
    if (ctx == nullptr || !ctx->last.valid || index < 0 ||
        index >= (int)ctx->last.prompts.size()) {
        return nullptr;
    }
    return ctx->last.prompts[index].c_str();
}

AICORE_CAPI int aicore_gkd_result_roi(const aicore_gkd_ctx* ctx,
                                      float out_bbox[4]) {
    if (ctx == nullptr || out_bbox == nullptr) return -1;
    std::memset(out_bbox, 0, sizeof(float) * 4);
    if (!ctx->last.valid) return -1;
    std::memcpy(out_bbox, ctx->last.roi, sizeof(ctx->last.roi));
    return 0;
}

AICORE_CAPI int aicore_gkd_last_timings(const aicore_gkd_ctx* ctx,
                                        aicore_gkd_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_timings) {
        return -1;
    }
    *out_timings = ctx->timings;
    return 0;
}

AICORE_CAPI int aicore_gkd_last_pipeline_timings(
        const aicore_gkd_ctx* ctx, aicore_pipeline_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_pipeline) {
        return -1;
    }
    *out_timings = ctx->pipeline;
    return 0;
}

AICORE_CAPI const char* aicore_gkd_context_model_name(aicore_gkd_ctx* ctx) {
    return ctx != nullptr && ctx->session != nullptr ? ctx->model_name.c_str()
                                                     : "";
}

AICORE_CAPI int32_t aicore_gkd_context_image_size(const aicore_gkd_ctx* ctx) {
    if (ctx == nullptr || ctx->session == nullptr) return 0;
    return ctx->session->params().img_size;
}

AICORE_CAPI const char* aicore_gkd_context_device(const aicore_gkd_ctx* ctx) {
    return ctx != nullptr ? ctx->resolved_device.c_str() : "";
}

AICORE_CAPI int aicore_gkd_context_threads(const aicore_gkd_ctx* ctx) {
    return ctx != nullptr ? ctx->threads : 0;
}

AICORE_CAPI char* aicore_gkd_info_json(aicore_gkd_ctx* ctx) {
    if (ctx == nullptr || ctx->session == nullptr) return nullptr;
    const gkd::ModelParams& P = ctx->session->params();
    std::ostringstream json;
    json << "{\"model\":\"" << json_escape(ctx->model_name) << "\"";
    json << ",\"device\":\"" << json_escape(ctx->resolved_device) << "\"";
    json << ",\"threads\":" << ctx->threads;
    json << ",\"image_size\":" << P.img_size;
    json << ",\"model_path\":\"" << json_escape(ctx->model_path) << "\"";
    json << ",\"result\":";
    if (ctx->last.valid) {
        json << "{\"roi\":[";
        json << ctx->last.roi[0] << "," << ctx->last.roi[1] << ",";
        json << ctx->last.roi[2] << "," << ctx->last.roi[3] << "]";
        json << ",\"keypoints\":[";
        for (int i = 0; i < ctx->last.n_keypoints; ++i) {
            const aicore_gkd_keypoint kp =
                    aicore_gkd_result_keypoint_at(ctx, i);
            if (i > 0) json << ",";
            json << "{\"x\":" << kp.x << ",\"y\":" << kp.y;
            json << ",\"x_norm\":" << kp.x_norm << ",\"y_norm\":" << kp.y_norm;
            json << ",\"score\":" << kp.score;
            const char* prompt = aicore_gkd_result_prompt_at(ctx, i);
            json << ",\"prompt\":";
            if (prompt != nullptr) {
                json << "\"" << json_escape(prompt) << "\"";
            } else {
                json << "null";
            }
            json << "}";
        }
        json << "]}";
    } else {
        json << "null";
    }
    json << "}";
    return dup_cstr(json.str());
}

AICORE_CAPI int aicore_gkd_load_path_rgb(const char* image_path,
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
    // Compatibility file-decode path (the typed hot path takes an
    // aicore_image_view; AICore links Qt::Gui for this wrapper only).
    QImage image(QString::fromUtf8(image_path));
    if (image.isNull()) return -1;
    aicore::capi::PackedRgb packed = aicore::capi::qimage_to_packed_rgb(image);
    if (packed.data == nullptr) return -1;
    *out_rgb = packed.data;
    *out_width = packed.width;
    *out_height = packed.height;
    return 0;
}

AICORE_CAPI int aicore_gkd_warmup_backend(const char* device) {
    return aicore_warmup_backend(device);
}

AICORE_CAPI void aicore_gkd_shutdown(void) { aicore_runtime_shutdown(); }

AICORE_CAPI char* aicore_gkd_model_cache_dir(void) {
    return dup_cstr(aicore::gkd_model_cache_dir());
}
