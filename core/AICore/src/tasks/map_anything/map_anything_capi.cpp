// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// MapAnything C API implementation over the in-tree ggml port
// (tasks/map_anything/: gguf loader + VGGTRuntime + official image
// preprocessing). The typed hot path is aicore_map_anything_run_images with
// borrowed aicore_image_view values; the frames and packed-RGB entry points
// are parity/compatibility wrappers around the same session.

#include "aicore/map_anything_capi.h"

#include <QImage>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
#include "tasks/map_anything/map_anything_backend.hpp"
#include "tasks/map_anything/map_anything_gguf_loader.hpp"
#include "tasks/map_anything/map_anything_graph.hpp"
#include "tasks/map_anything/map_anything_graph_builder.hpp"
#include "tasks/map_anything/map_anything_image_io.hpp"

namespace {

using aicore::capi::dup_cstr;
using aicore::capi::json_escape;

double steady_ms() {
    return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

map_anything::LogLevel log_level_from_int(int level) {
    if (level <= 0) return map_anything::LogLevel::Debug;
    if (level == 1) return map_anything::LogLevel::Info;
    if (level == 2) return map_anything::LogLevel::Warn;
    return map_anything::LogLevel::Error;
}

int32_t pose_stride_for(const std::string& architecture) {
    // Layout contract (see aicore/map_anything_capi.h):
    //   vggt_omega / vggt      -> pose_enc [n*9]
    //   pi3 / pi3x / mapanything -> camera-to-world 4x4 [n*16]
    //   dust3r                 -> no pose head
    if (architecture == "pi3" || architecture == "pi3x" ||
        architecture == "mapanything") {
        return 16;
    }
    if (architecture == "dust3r") return 0;
    return 9;
}

}  // namespace

// Options/context layouts matching the opaque typedefs in
// aicore/map_anything_capi.h. Global scope (not anonymous) — the C header
// already declares the tags.
struct aicore_map_anything_options {
    aicore::capi::CommonOptions common;
    int image_size = 0;  // 0 = checkpoint's nominal resolution
    std::string resize_mode = "balanced";
    int log_level = 1;
    std::string dump_dir;
};

struct aicore_map_anything_ctx {
    std::unique_ptr<map_anything::GGUFModel> model;
    map_anything::Backend backend;
    std::unique_ptr<map_anything::VGGTRuntime> runtime;
    int cur_s = 0, cur_h = 0, cur_w = 0;

    std::string model_path;
    std::string device_request;
    std::string resolved_device;
    int32_t threads = 0;
    int image_size = 0;  // resolved option snapshot
    std::string resize_mode = "balanced";
    std::string dump_dir;
    std::string last_error;

    // Context-owned result store of the most recent successful run. The
    // public accessors return borrowed views into these members; they stay
    // valid until the next run call or aicore_map_anything_free.
    bool has_result = false;
    int32_t r_views = 0, r_height = 0, r_width = 0;
    int32_t r_pose_stride = 0;
    std::string architecture;
    // Per-view RESIZED content size before the centered white pad (the
    // frames path fills the full grid: no pad there).
    std::vector<std::pair<int, int>> r_valid;
    map_anything::VGGTOutputs out;
    aicore_pipeline_timings pipeline{};
    bool has_pipeline = false;
};

namespace {

// Ensures a runtime exists for (S, H, W). The rebuild re-stages the weights,
// so resolution sweeps pay the staging cost per new geometry. The backend
// lease lock must be held by the caller.
bool ensure_runtime(aicore_map_anything_ctx* ctx, int s, int h, int w) {
    if (ctx->runtime && ctx->cur_s == s && ctx->cur_h == h && ctx->cur_w == w) {
        return true;
    }
    ctx->runtime.reset();
    ctx->cur_s = ctx->cur_h = ctx->cur_w = 0;
    map_anything::RuntimeOptions opts;
    opts.dump_dir = ctx->dump_dir;
    ctx->runtime.reset(new map_anything::VGGTRuntime(*ctx->model, ctx->backend,
                                                     s, h, w, opts));
    if (!ctx->runtime->valid()) {
        ctx->runtime.reset();
        ctx->last_error =
                "runtime build failed for the requested geometry (see the "
                "[MapAnything] log lines)";
        return false;
    }
    ctx->cur_s = s;
    ctx->cur_h = h;
    ctx->cur_w = w;
    return true;
}

#ifdef MAP_ANYTHING_ENABLE_DUMP
// Debug-only preprocessing boundary: the external CLI consumes this exact
// (S, 3, H, W) f32 payload through --bin, isolating graph parity from decode
// and resize parity without extending the public C ABI.
bool dump_preprocessed_frames(aicore_map_anything_ctx* ctx,
                              const std::vector<float>& frames) {
    if (ctx->dump_dir.empty()) return true;
    const std::filesystem::path path =
            std::filesystem::path(ctx->dump_dir) / "cpp_frames_f32.bin";
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        ctx->last_error = "cannot open preprocessing dump: " + path.string();
        return false;
    }
    file.write(reinterpret_cast<const char*>(frames.data()),
               static_cast<std::streamsize>(frames.size() * sizeof(float)));
    if (!file) {
        ctx->last_error = "cannot write preprocessing dump: " + path.string();
        return false;
    }
    return true;
}
#endif

// Shared tail of both run entry points: executes the graph and snapshots the
// typed result into the context-owned store. The backend lease lock is held
// for the whole ensure + compute scope (the physical device handle is shared
// process-wide).
int run_common(aicore_map_anything_ctx* ctx,
               std::vector<float>& frames,
               int s,
               int h,
               int w,
               double preprocess_ms,
               double t_entry) {
    if (aicore_cancel_requested()) {
        ctx->last_error = "cancelled";
        return -2;
    }
    auto lease_lock = ctx->backend.lease.lock();
    if (!ensure_runtime(ctx, s, h, w)) {
        return -2;
    }
    map_anything::VGGTOutputs out;
    const double t_inf0 = steady_ms();
    if (!ctx->runtime->run(frames.data(), out)) {
        ctx->last_error = "inference failed (see the [MapAnything] log lines)";
        return -2;
    }
    const double inference_ms = steady_ms() - t_inf0;

    ctx->out = std::move(out);
    ctx->architecture = ctx->model->meta.architecture;
    ctx->r_views = s;
    ctx->r_height = ctx->out.out_h > 0 ? ctx->out.out_h : h;
    ctx->r_width = ctx->out.out_w > 0 ? ctx->out.out_w : w;
    ctx->r_pose_stride = pose_stride_for(ctx->architecture);
    // Single choke point: keep per-view valid sizes consistent with the
    // result grid (run_images filled the pre-pad sizes; the frames/packed
    // paths have no pad, so the valid rect is the full grid).
    if ((int)ctx->r_valid.size() != s) ctx->r_valid.assign(s, {w, h});
    ctx->has_result = true;

    const double e2e_ms = steady_ms() - t_entry;
    ctx->pipeline.abi_version = AICORE_PIPELINE_TIMINGS_ABI_VERSION;
    ctx->pipeline.valid_fields = AICORE_TIMING_PREPROCESS |
                                 AICORE_TIMING_INFERENCE |
                                 AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    ctx->pipeline.preprocess_ms = preprocess_ms;
    ctx->pipeline.inference_ms = inference_ms;
    ctx->pipeline.postprocess_ms =
            std::max(0.0, e2e_ms - preprocess_ms - inference_ms);
    ctx->pipeline.e2e_ms = e2e_ms;
    ctx->has_pipeline = true;
    ctx->last_error.clear();
    return 0;
}

}  // namespace

AICORE_CAPI int aicore_map_anything_abi_version(void) { return 1; }

AICORE_CAPI aicore_map_anything_options* aicore_map_anything_options_new(void) {
    return new (std::nothrow) aicore_map_anything_options();
}

AICORE_CAPI void aicore_map_anything_options_free(
        aicore_map_anything_options* opts) {
    delete opts;
}

AICORE_CAPI void aicore_map_anything_options_set_device(
        aicore_map_anything_options* opts, const char* device) {
    if (opts != nullptr) aicore::capi::set_device(opts->common, device);
}

AICORE_CAPI void aicore_map_anything_options_set_threads(
        aicore_map_anything_options* opts, int n_threads) {
    if (opts != nullptr) aicore::capi::set_threads(opts->common, n_threads);
}

AICORE_CAPI void aicore_map_anything_options_set_image_size(
        aicore_map_anything_options* opts, int image_size) {
    if (opts == nullptr || image_size < 0) return;
    opts->image_size = image_size;  // 0 = model default
}

AICORE_CAPI void aicore_map_anything_options_set_resize_mode(
        aicore_map_anything_options* opts, const char* mode) {
    if (opts == nullptr || mode == nullptr) return;
    const std::string want(mode);
    if (want != "balanced" && want != "max_size") return;
    opts->resize_mode = want;
}

AICORE_CAPI void aicore_map_anything_options_set_log_level(
        aicore_map_anything_options* opts, int log_level) {
    if (opts != nullptr) opts->log_level = log_level;
}

AICORE_CAPI void aicore_map_anything_options_set_dump_dir(
        aicore_map_anything_options* opts, const char* dir) {
    if (opts != nullptr) opts->dump_dir = dir != nullptr ? dir : "";
}

AICORE_CAPI aicore_map_anything_ctx* aicore_map_anything_load_opts(
        const char* gguf_path, const aicore_map_anything_options* opts) {
    if (gguf_path == nullptr) return nullptr;
    auto* ctx = new (std::nothrow) aicore_map_anything_ctx();
    if (ctx == nullptr) return nullptr;

    ctx->model_path = gguf_path;
    ctx->device_request = opts != nullptr && !opts->common.device.empty()
                                  ? opts->common.device
                                  : "auto";
    ctx->threads = opts != nullptr ? opts->common.threads : 0;
    ctx->image_size = opts != nullptr ? opts->image_size : 0;
    ctx->resize_mode = opts != nullptr && !opts->resize_mode.empty()
                               ? opts->resize_mode
                               : "balanced";
    ctx->dump_dir = opts != nullptr ? opts->dump_dir : "";
    map_anything::set_log_level(
            log_level_from_int(opts != nullptr ? opts->log_level : 1));

    // VRAM admission guard (same contract as gkd/yolo: a fatal backend
    // abort under memory pressure at inference time turns into a clean,
    // actionable load error here).
    {
        std::error_code size_ec;
        const auto model_bytes = std::filesystem::file_size(gguf_path, size_ec);
        if (!size_ec &&
            !ggml_common::gpu_admission_check(ctx->device_request,
                                              static_cast<size_t>(model_bytes),
                                              &ctx->last_error)) {
            return ctx;  // not ready; last_error carries the reason
        }
    }

    ctx->backend = map_anything::init_backend(ctx->device_request, ctx->threads,
                                              &ctx->last_error);
    if (!ctx->backend.valid()) {
        if (ctx->last_error.empty()) {
            ctx->last_error = "no compute backend available";
        }
        return ctx;
    }
    ctx->threads = ctx->backend.n_threads;
    ctx->resolved_device =
            ctx->backend.name == "CPU" ? std::string("cpu") : ctx->backend.name;

    ctx->model = map_anything::load_gguf(gguf_path);
    if (!ctx->model) {
        ctx->last_error = "failed to load GGUF: " + std::string(gguf_path) +
                          " (see the [MapAnything] log lines for the reason)";
        return ctx;
    }
    // Early architecture check: create_builder logs the registered keys when
    // the GGUF's general.architecture is unknown.
    auto probe = map_anything::create_builder(ctx->model->meta.architecture);
    if (!probe) {
        ctx->last_error =
                "unknown architecture '" + ctx->model->meta.architecture + "'";
        return ctx;
    }
    ctx->last_error.clear();
    return ctx;
}

AICORE_CAPI void aicore_map_anything_free(aicore_map_anything_ctx* ctx) {
    delete ctx;
}

AICORE_CAPI int aicore_map_anything_is_ready(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->model != nullptr && ctx->backend.valid() ? 1
                                                                           : 0;
}

AICORE_CAPI const char* aicore_map_anything_last_error(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && !ctx->last_error.empty() ? ctx->last_error.c_str()
                                                      : nullptr;
}

AICORE_CAPI void aicore_map_anything_free_buffer(void* p) { std::free(p); }

AICORE_CAPI int aicore_map_anything_load_path_rgb(const char* image_path,
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

AICORE_CAPI int aicore_map_anything_run_images(aicore_map_anything_ctx* ctx,
                                               const aicore_image_view* views,
                                               int32_t n_views) {
    if (ctx == nullptr) return -1;
    ctx->last_error.clear();
    if (!ctx->model || !ctx->backend.valid()) {
        ctx->last_error = "no loaded model";
        return -1;
    }
    if (views == nullptr || n_views <= 0) {
        ctx->last_error = "run_images: invalid arguments";
        return -1;
    }
    for (int32_t i = 0; i < n_views; ++i) {
        if (views[i].data == nullptr || views[i].width <= 0 ||
            views[i].height <= 0) {
            ctx->last_error = "run_images: invalid image view";
            return -1;
        }
    }
    const double t0 = steady_ms();
    const int patch = ctx->model->meta.patch_size;
    int image_size = ctx->image_size;
    if (image_size <= 0) {
        // Checkpoint's nominal resolution (dust3r ships 512_dpt; the omega
        // variants name their nominal resolution; everything else gates at
        // 518) — same contract as the upstream C API.
        image_size =
                ctx->model->meta.architecture == "dust3r" ||
                                ctx->model->meta.architecture == "vggt_omega"
                        ? 512
                        : 518;
    }
    if (patch > 0 && image_size % patch != 0) {
        // Keep the official invariant: resolution divisible by patch.
        image_size = (image_size / patch) * patch;
        if (image_size < patch) image_size = patch;
    }
    std::vector<float> frames;
    int h = 0, w = 0;
    if (!map_anything::load_images_official_views(
                views, n_views, image_size, patch, frames, h, w,
                ctx->resize_mode, &ctx->r_valid)) {
        ctx->last_error = "image preprocessing failed";
        return -1;
    }
#ifdef MAP_ANYTHING_ENABLE_DUMP
    if (!dump_preprocessed_frames(ctx, frames)) return -1;
#endif
    const double preprocess_ms = steady_ms() - t0;
    return run_common(ctx, frames, n_views, h, w, preprocess_ms, t0);
}

AICORE_CAPI int aicore_map_anything_run_frames(aicore_map_anything_ctx* ctx,
                                               const float* frames_f32,
                                               int32_t n_views,
                                               int32_t height,
                                               int32_t width) {
    if (ctx == nullptr) return -1;
    ctx->last_error.clear();
    if (!ctx->model || !ctx->backend.valid()) {
        ctx->last_error = "no loaded model";
        return -1;
    }
    if (frames_f32 == nullptr || n_views <= 0 || height <= 0 || width <= 0) {
        ctx->last_error = "run_frames: invalid arguments";
        return -1;
    }
    const double t0 = steady_ms();
    std::vector<float> frames(
            frames_f32,
            frames_f32 + (size_t)n_views * 3 * (size_t)height * (size_t)width);
    // Frames arrive preprocessed; only the input validation is attributable
    // to the preprocess stage (truthful zero would under-report e2e).
    return run_common(ctx, frames, n_views, height, width, steady_ms() - t0,
                      t0);
}

AICORE_CAPI int32_t
aicore_map_anything_result_view_count(const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result ? ctx->r_views : 0;
}

AICORE_CAPI int32_t
aicore_map_anything_result_height(const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result ? ctx->r_height : 0;
}

AICORE_CAPI int32_t
aicore_map_anything_result_width(const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result ? ctx->r_width : 0;
}

AICORE_CAPI const char* aicore_map_anything_result_architecture(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result ? ctx->architecture.c_str() : "";
}

AICORE_CAPI int aicore_map_anything_result_view_valid_size(
        const aicore_map_anything_ctx* ctx,
        int32_t view,
        int32_t* w,
        int32_t* h) {
    if (ctx == nullptr || !ctx->has_result || w == nullptr || h == nullptr ||
        view < 0 || view >= ctx->r_views) {
        return -1;
    }
    if ((int)ctx->r_valid.size() != ctx->r_views) {
        *w = ctx->r_width;
        *h = ctx->r_height;
        return 0;
    }
    *w = ctx->r_valid[view].first;
    *h = ctx->r_valid[view].second;
    return 0;
}

AICORE_CAPI const float* aicore_map_anything_result_depth(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result && !ctx->out.depth.empty()
                   ? ctx->out.depth.data()
                   : nullptr;
}

AICORE_CAPI const float* aicore_map_anything_result_conf(
        const aicore_map_anything_ctx* ctx) {
    if (ctx == nullptr || !ctx->has_result) return nullptr;
    // vggt family depth_conf vs pi3/dust3r conf — both surface through the
    // same accessor with the semantics documented in the header.
    if (!ctx->out.depth_conf.empty()) return ctx->out.depth_conf.data();
    if (!ctx->out.conf.empty()) return ctx->out.conf.data();
    return nullptr;
}

AICORE_CAPI const float* aicore_map_anything_result_local_points(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result && !ctx->out.local_points.empty()
                   ? ctx->out.local_points.data()
                   : nullptr;
}

AICORE_CAPI const float* aicore_map_anything_result_points(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result && !ctx->out.world_points.empty()
                   ? ctx->out.world_points.data()
                   : nullptr;
}

AICORE_CAPI const float* aicore_map_anything_result_mask(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result && !ctx->out.mask.empty()
                   ? ctx->out.mask.data()
                   : nullptr;
}

AICORE_CAPI const float* aicore_map_anything_result_scale(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result && ctx->out.metric_scale > 0.0f
                   ? &ctx->out.metric_scale
                   : nullptr;
}

AICORE_CAPI const float* aicore_map_anything_result_pose(
        const aicore_map_anything_ctx* ctx) {
    if (ctx == nullptr || !ctx->has_result) return nullptr;
    if (ctx->r_pose_stride == 16) {
        return ctx->out.camera_poses.empty() ? nullptr
                                             : ctx->out.camera_poses.data();
    }
    if (ctx->r_pose_stride == 9) {
        return ctx->out.pose_enc.empty() ? nullptr : ctx->out.pose_enc.data();
    }
    return nullptr;
}

AICORE_CAPI int32_t
aicore_map_anything_result_pose_stride(const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->has_result ? ctx->r_pose_stride : 0;
}

AICORE_CAPI const float* aicore_map_anything_result_text_embedding(
        const aicore_map_anything_ctx* ctx, int32_t* out_dim) {
    if (out_dim != nullptr) *out_dim = 0;
    if (ctx == nullptr || !ctx->has_result || ctx->out.text_embedding.empty()) {
        return nullptr;
    }
    if (out_dim != nullptr) {
        *out_dim = (int32_t)ctx->out.text_embedding.size();
    }
    return ctx->out.text_embedding.data();
}

AICORE_CAPI int aicore_map_anything_last_pipeline_timings(
        const aicore_map_anything_ctx* ctx,
        aicore_pipeline_timings* out_timings) {
    if (ctx == nullptr || out_timings == nullptr || !ctx->has_pipeline) {
        return -1;
    }
    *out_timings = ctx->pipeline;
    return 0;
}

AICORE_CAPI int32_t
aicore_map_anything_context_patch_size(aicore_map_anything_ctx* ctx) {
    return ctx != nullptr && ctx->model ? ctx->model->meta.patch_size : 0;
}

AICORE_CAPI int32_t
aicore_map_anything_context_default_image_size(aicore_map_anything_ctx* ctx) {
    if (ctx == nullptr || !ctx->model) return 0;
    // Same dispatch as the typed run path (see aicore_map_anything_capi.h).
    if (ctx->model->meta.architecture == "dust3r") return 512;
    if (ctx->model->meta.architecture == "vggt_omega") return 512;
    return 518;
}

AICORE_CAPI const char* aicore_map_anything_context_device(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr ? ctx->resolved_device.c_str() : "";
}

AICORE_CAPI int aicore_map_anything_context_threads(
        const aicore_map_anything_ctx* ctx) {
    return ctx != nullptr ? ctx->threads : 0;
}

AICORE_CAPI char* aicore_map_anything_info_json(aicore_map_anything_ctx* ctx) {
    if (ctx == nullptr) return nullptr;
    std::ostringstream json;
    json << "{\"architecture\":\""
         << json_escape(ctx->model ? ctx->architecture : std::string()) << "\"";
    json << ",\"device\":\"" << json_escape(ctx->resolved_device) << "\"";
    json << ",\"threads\":" << ctx->threads;
    json << ",\"patch_size\":"
         << (ctx->model ? ctx->model->meta.patch_size : 0);
    json << ",\"default_image_size\":"
         << aicore_map_anything_context_default_image_size(ctx);
    json << ",\"image_size\":" << ctx->image_size;
    json << ",\"resize_mode\":\"" << json_escape(ctx->resize_mode) << "\"";
    json << ",\"model_path\":\"" << json_escape(ctx->model_path) << "\"";
    json << ",\"result\":";
    if (ctx->has_result) {
        json << "{\"views\":" << ctx->r_views;
        json << ",\"height\":" << ctx->r_height;
        json << ",\"width\":" << ctx->r_width;
        json << ",\"pose_stride\":" << ctx->r_pose_stride;
        json << ",\"has_points\":"
             << (ctx->out.world_points.empty() ? "false" : "true");
        json << ",\"has_local_points\":"
             << (ctx->out.local_points.empty() ? "false" : "true");
        json << ",\"has_mask\":" << (ctx->out.mask.empty() ? "false" : "true");
        json << ",\"e2e_ms\":" << ctx->pipeline.e2e_ms;
        json << "}";
    } else {
        json << "null";
    }
    json << "}";
    return dup_cstr(json.str());
}

AICORE_CAPI int aicore_map_anything_warmup_backend(const char* device) {
    return aicore_warmup_backend(device);
}

AICORE_CAPI void aicore_map_anything_shutdown(void) {
    aicore_runtime_shutdown();
}

AICORE_CAPI char* aicore_map_anything_model_cache_dir(void) {
    return dup_cstr(aicore::map_anything_model_cache_dir());
}
