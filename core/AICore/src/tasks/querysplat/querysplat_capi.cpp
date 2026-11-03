// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// QuerySplat C API implementation over the in-tree port. Mirrors the
// map_anything capi contract: options snapshots, a context-owned result
// store handed out as borrowed views, last_error on the context, common
// pipeline timings, cancellation checks, and the backend lease held across
// the whole encoder -> camera math -> decoder -> rasterizer scope. The
// encoder is the in-tree map_anything VGGT-Omega builder through the
// EncoderRuntime facade (trellis -> rmbg delegation pattern); no second
// encoder implementation exists in this tree.
#include "aicore/querysplat_capi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "aicore/map_anything_capi.h"
#include "aicore/runtime_capi.h"
#include "common/capi_utils.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "map_anything_encoder_runtime.hpp"
#include "map_anything_gguf_loader.hpp"
#include "querysplat_camera_math.hpp"
#include "querysplat_graph.hpp"
#include "querysplat_log.hpp"
#include "querysplat_lpips_graph.hpp"
#include "querysplat_ply.hpp"
#include "querysplat_preprocess.hpp"
#include "querysplat_rasterizer.hpp"
#include "querysplat_tto_loss.hpp"

namespace {

// Thread-local log tag guard for the public C entries: the shared
// map-anything encoder runtime and gguf loader log through
// map_anything::log_*, and every line emitted while a querysplat context
// is loading/running must carry the [querysplat] identity, not the
// runtime's default tag. RAII restores the default on every exit path.
struct ScopedTaskLogTag {
    ScopedTaskLogTag() { map_anything::set_log_tag("[querysplat] "); }
    ~ScopedTaskLogTag() { map_anything::set_log_tag(nullptr); }
    ScopedTaskLogTag(const ScopedTaskLogTag&) = delete;
    ScopedTaskLogTag& operator=(const ScopedTaskLogTag&) = delete;
};

}  // namespace

using aicore::capi::dup_cstr;
using aicore::capi::image_view_to_packed_rgb;

// Options/context layouts matching the opaque typedefs in
// aicore/querysplat_capi.h. Global scope (not anonymous) — the C header
// already declares the tags.
struct aicore_querysplat_options {
    aicore::capi::CommonOptions common;
    int f32_accuracy = 1;  // torch-parity profile (see the public header)
    int image_height = 0;  // 0,0 = decoder GGUF metadata resolution
    int image_width = 0;
    float bg_color[3] = {0.5f, 0.5f, 0.5f};
    int tto_steps = 0;  // options-level TTO default; run_tto takes steps
    double tto_lr = 5e-3;
    double tto_lpips_weight = 0.05;
    float opacity_threshold = 0.05f;
    int log_level = 1;
    std::string dump_dir;
};

// One context: encoder facade + decoder graph + the context-owned result
// store. The BackendLease lives inside the EncoderRuntime's Backend; the
// decoder graph and rasterizer run on the same physical handle.
struct aicore_querysplat_ctx {
    map_anything::EncoderRuntime enc;
    std::unique_ptr<qs::QuerySplatDecoder> dec;
    qs::DecoderConfig meta_cfg;  // decoder GGUF metadata (resolution, layout)

    // option snapshot (resolved at load; info JSON reports it)
    std::string encoder_path;
    std::string decoder_path;
    std::string device_request;
    std::string resolved_device;
    int32_t threads = 0;
    int f32_accuracy = 1;
    int image_height = 0, image_width = 0;
    float bg_color[3] = {0.5f, 0.5f, 0.5f};
    double tto_lr = 5e-3;
    double tto_lpips_weight = 0.05;
    float opacity_threshold = 0.05f;
    int log_level = 1;
    std::string dump_dir;
    std::string last_error;

    // The frame/camera state of the most recent successful run. TTO and the
    // renderers re-consume them; every buffer is context-owned and handed
    // out as borrowed views.
    bool has_result = false;
    int32_t r_views = 0, r_height = 0, r_width = 0;
    std::vector<float> r_frames;  // (S,3,H,W) normalized rgb
    // (S,3,H,W) [0,1] sRGB planes — the pre-normalization frames. The TTO
    // loss targets and the depth-pointcloud colors consume THESE (upstream
    // cli.cpp passes `frames` [0,1] to the loss and `rgb_norm` only to
    // tto_begin's in-forward patch_embed); normalizing the targets instead
    // makes the optimizer chase an unreachable image (observed: L1 rising
    // 0.75 -> 0.91, negative SSIM).
    std::vector<float> r_frames_01;
    std::vector<float> r_plucker;       // (S,6,H,W)
    qs::CameraData r_cams;              // w2c / c2w / intrinsics
    std::vector<float> r_pose_store;    // (S,9) copy of the encoder output
    map_anything::VGGTOutputs enc_out;  // depth / depth_conf / tokens
    qs::DecoderOutputs dec_out;         // gaussians
    // render store (most recent aicore_querysplat_render_views / TTO output)
    qs::RenderedViews render_out;
    std::vector<unsigned char> render_rgb8;
    // TTO store
    std::vector<aicore_querysplat_tto_step> tto_losses;
    std::vector<float> tto_gaussians_store;
    // depth-point-cloud store
    std::vector<float> pc_xyz;
    std::vector<unsigned char> pc_rgb;
    aicore_pipeline_timings pipeline{};
    bool has_pipeline = false;
};

namespace {

constexpr int kRunOutOfMemory = -3;  // see aicore_querysplat_run_images

double steady_ms() {
    return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now().time_since_epoch())
            .count();
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

}  // namespace

// Shared inference tail: encoder -> camera math -> decoder. The backend
// lease lock is held across the whole scope (one physical device handle).
int run_common(aicore_querysplat_ctx* ctx,
               std::vector<float>&& frames,
               int n,
               int h,
               int w,
               double preprocess_ms,
               double t_entry) {
    if (aicore_cancel_requested()) {
        ctx->last_error = "cancelled";
        return -2;
    }
    if (frames.size() != (size_t)n * 3 * h * w) {
        ctx->last_error = "internal: frame buffer size mismatch";
        return -1;
    }
    auto lease_lock = ctx->enc.backend().lease.lock();

    std::string err;
    if (!ctx->enc.ensure_geometry(n, h, w, &err)) {
        ctx->last_error = err.empty() ? "encoder runtime build failed (see the "
                                        "[MapAnything] log lines)"
                                      : err;
        return ctx->enc.backend().valid() ? -2 : kRunOutOfMemory;
    }
    const double t_inf0 = steady_ms();
    if (!ctx->enc.run(frames.data(), ctx->enc_out, &err)) {
        ctx->last_error =
                err.empty() ? "encoder run failed (see the log lines)" : err;
        return -2;
    }
    // host camera math (official stage: camera_math)
    ctx->r_cams = qs::decode_cameras(ctx->enc_out.pose_enc.data(), n, h, w);
    ctx->r_plucker = qs::plucker_from_cameras(ctx->r_cams, h, w);
    ctx->r_frames_01.assign(frames.begin(), frames.end());
    ctx->r_frames = qs::imagenet_normalize(frames.data(), n, frames.size());
    if (!ctx->dec->ensure_shape(n, h, w, ctx->enc_out.patch_token_start)) {
        ctx->last_error = "decoder graph build failed for the geometry";
        return -2;
    }
    if (!ctx->dec->run(ctx->enc_out.cached_tokens.data(), ctx->r_frames.data(),
                       ctx->r_plucker.data(), ctx->dec_out)) {
        ctx->last_error = "decoder run failed (see the [QuerySplat] lines)";
        return -2;
    }
    const double inference_ms = steady_ms() - t_inf0;

    ctx->r_pose_store = ctx->enc_out.pose_enc;
    ctx->r_views = n;
    ctx->r_height = h;
    ctx->r_width = w;
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

// TTO loop, ported from the upstream cli.cpp --use_tto block (the official
// _optimize_memory). Requires a seeded session and consumes ctx->r_*.
int run_tto_loop(aicore_querysplat_ctx* ctx,
                 int n_steps,
                 std::unique_ptr<qs::LpipsVgg>& lpips) {
    const int S = ctx->r_views;
    const int H = ctx->r_height;
    const int W = ctx->r_width;
    const qs::DecoderConfig& cfg = ctx->meta_cfg;
    const double lw = ctx->tto_lpips_weight;
    const int stride = 11 + ctx->dec_out.color_dim;
    const int post_out_dim =
            3 * (cfg.gaussian_sh_degree + 1) * (cfg.gaussian_sh_degree + 1) + 1;

    if (!ctx->dec->tto_begin(S, H, W, ctx->enc_out.patch_token_start,
                             ctx->enc_out.cached_tokens.data(),
                             ctx->r_frames.data(), ctx->r_plucker.data(),
                             ctx->tto_lr)) {
        ctx->last_error =
                "TTO session init failed (sh>0 checkpoints are "
                "unsupported by the official TTO path)";
        return -2;
    }
    qs::RenderConfig rcfg;
    rcfg.width = W;
    rcfg.height = H;
    rcfg.near_plane = cfg.znear;
    rcfg.far_plane = cfg.zfar;
    rcfg.sh_degree = cfg.gaussian_sh_degree;
    std::memcpy(rcfg.bg_color, ctx->bg_color, sizeof(rcfg.bg_color));

    ctx->tto_losses.clear();
    ctx->tto_losses.reserve(n_steps);
    for (int step = 1; step <= n_steps; step++) {
        if (aicore_cancel_requested()) {
            ctx->last_error = "cancelled";
            return -2;
        }
        std::vector<float> deconv_raw, color_raw;
        if (!ctx->dec->tto_forward(deconv_raw, color_raw)) {
            ctx->last_error = "TTO forward failed";
            return -2;
        }
        qs::DecoderOutputs to =
                ctx->dec->tto_gaussians_from(deconv_raw, color_raw);
        qs::GaussianOutput g;
        g.N = to.N;
        g.color_dim = to.color_dim;
        g.data = std::move(to.gaussians);
        g.opacity.resize((size_t)g.N);
        for (int64_t i = 0; i < g.N; i++)
            g.opacity[(size_t)i] = g.data[(size_t)i * stride + 3];
        qs::RenderedViews views =
                qs::render_gaussians(g, ctx->r_cams.w2c.data(),
                                     ctx->r_cams.intrinsics.data(), S, rcfg);
        // LPIPS first (its grad feeds the combined loss backward)
        std::vector<float> d_lpips;
        double lp_val = 0.0;
        if (lw > 0 && lpips) {
            std::vector<float> lv;
            if (!lpips->value_and_grad(views.images.data(),
                                       ctx->r_frames_01.data(), S, H, W, &lv,
                                       &d_lpips)) {
                ctx->last_error = "LPIPS value_and_grad failed";
                return -2;
            }
            lp_val = lv.empty() ? 0.0 : (double)lv[0];
        }
        qs::TtoLossOut lo;
        if (!qs::tto_loss_value_and_grad(
                    views.images.data(), ctx->r_frames_01.data(), S, H, W,
                    (float)lw, lw > 0 ? d_lpips.data() : nullptr, &lo)) {
            ctx->last_error = "TTO loss failed";
            return -2;
        }
        qs::GaussianGrads gg;
        if (!qs::render_backward(g, ctx->r_cams.w2c.data(),
                                 ctx->r_cams.intrinsics.data(), S, rcfg, views,
                                 lo.d_pred.data(), &gg)) {
            ctx->last_error = "render backward failed";
            return -2;
        }
        // pack the gaussian grads into the raw layout (N, 11+color_dim)
        std::vector<float> dg((size_t)gg.N * stride);
        for (int64_t i = 0; i < gg.N; i++) {
            float* r = &dg[(size_t)i * stride];
            for (int c = 0; c < 3; c++) r[c] = gg.pos[(size_t)i * 3 + c];
            r[3] = gg.opacity[(size_t)i];
            for (int c = 0; c < 3; c++) r[4 + c] = gg.scale[(size_t)i * 3 + c];
            for (int c = 0; c < 4; c++) r[7 + c] = gg.rot[(size_t)i * 4 + c];
            for (int c = 0; c < gg.color_dim; c++)
                r[11 + c] = gg.color[(size_t)i * gg.color_dim + c];
        }
        std::vector<float> dgeom((size_t)cfg.num_gs_tokens * 10 *
                                 cfg.num_gaussians_per_token),
                dcol((size_t)cfg.num_gs_tokens * post_out_dim *
                     cfg.num_gaussians_per_token);
        qs::activate_gaussians_backward(
                deconv_raw.data(), color_raw.data(), dg.data(),
                cfg.num_gs_tokens, cfg.num_gaussians_per_token, post_out_dim,
                cfg.gaussian_scale_cap, cfg.gaussian_z_offset,
                cfg.gaussian_sh_degree, cfg.gaussian_sh_rest_scale,
                dgeom.data(), dcol.data());
        if (!ctx->dec->tto_backward(dgeom.data(), dcol.data(), step - 1)) {
            ctx->last_error = "TTO backward failed";
            return -2;
        }
        aicore_querysplat_tto_step row{};
        row.l1 = lo.l1;
        row.ssim = lo.ssim;
        row.lpips = lp_val;
        ctx->tto_losses.push_back(row);
        qs::log_info("TTO [%d/%d] L1=%.6f SSIM=%.6f LPIPS=%.6f", step, n_steps,
                     lo.l1, lo.ssim, lp_val);
    }
    // optimized gaussians (POST-step decode, the official capture())
    if (!ctx->dec->tto_gaussians(ctx->dec_out)) {
        ctx->last_error = "TTO final decode failed";
        return -2;
    }
    return 0;
}

AICORE_CAPI int aicore_querysplat_abi_version(void) { return 1; }

AICORE_CAPI aicore_querysplat_options* aicore_querysplat_options_new(void) {
    return new (std::nothrow) aicore_querysplat_options();
}

AICORE_CAPI void aicore_querysplat_options_free(
        aicore_querysplat_options* opts) {
    delete (aicore_querysplat_options*)opts;
}

AICORE_CAPI void aicore_querysplat_options_set_device(
        aicore_querysplat_options* opts, const char* device) {
    if (opts)
        aicore::capi::set_device(((aicore_querysplat_options*)opts)->common,
                                 device);
}

AICORE_CAPI void aicore_querysplat_options_set_threads(
        aicore_querysplat_options* opts, int n_threads) {
    if (opts)
        aicore::capi::set_threads(((aicore_querysplat_options*)opts)->common,
                                  n_threads);
}

AICORE_CAPI void aicore_querysplat_options_set_f32_accuracy(
        aicore_querysplat_options* opts, int enable) {
    if (opts && enable >= 0)
        ((aicore_querysplat_options*)opts)->f32_accuracy = enable ? 1 : 0;
}

AICORE_CAPI void aicore_querysplat_options_set_image_size(
        aicore_querysplat_options* opts, int height, int width) {
    if (!opts || height < 0 || width < 0) return;
    ((aicore_querysplat_options*)opts)->image_height = height;
    ((aicore_querysplat_options*)opts)->image_width = width;
}

AICORE_CAPI void aicore_querysplat_options_set_bg_color(
        aicore_querysplat_options* opts, float r, float g, float b) {
    if (!opts) return;
    ((aicore_querysplat_options*)opts)->bg_color[0] = r;
    ((aicore_querysplat_options*)opts)->bg_color[1] = g;
    ((aicore_querysplat_options*)opts)->bg_color[2] = b;
}

AICORE_CAPI void aicore_querysplat_options_set_tto_steps(
        aicore_querysplat_options* opts, int steps) {
    if (opts && steps >= 0)
        ((aicore_querysplat_options*)opts)->tto_steps = steps;
}

AICORE_CAPI void aicore_querysplat_options_set_tto_lr(
        aicore_querysplat_options* opts, double lr) {
    if (opts && lr > 0) ((aicore_querysplat_options*)opts)->tto_lr = lr;
}

AICORE_CAPI void aicore_querysplat_options_set_tto_lpips_weight(
        aicore_querysplat_options* opts, double weight) {
    if (opts && weight >= 0)
        ((aicore_querysplat_options*)opts)->tto_lpips_weight = weight;
}

AICORE_CAPI void aicore_querysplat_options_set_opacity_threshold(
        aicore_querysplat_options* opts, float threshold) {
    if (opts && threshold >= 0.f && threshold <= 1.f)
        ((aicore_querysplat_options*)opts)->opacity_threshold = threshold;
}

AICORE_CAPI void aicore_querysplat_options_set_log_level(
        aicore_querysplat_options* opts, int log_level) {
    if (opts && log_level >= 0 && log_level <= 3)
        ((aicore_querysplat_options*)opts)->log_level = log_level;
}

AICORE_CAPI void aicore_querysplat_options_set_dump_dir(
        aicore_querysplat_options* opts, const char* dir) {
    if (opts) ((aicore_querysplat_options*)opts)->dump_dir = dir ? dir : "";
}

AICORE_CAPI aicore_querysplat_ctx* aicore_querysplat_load_opts(
        const char* encoder_gguf,
        const char* decoder_gguf,
        const aicore_querysplat_options* opts) {
    ScopedTaskLogTag taskLogTag;
    aicore_querysplat_options defaults;
    const aicore_querysplat_options& o =
            opts ? *(const aicore_querysplat_options*)opts : defaults;

    // resolve catalog defaults for the paths
    std::string enc_path = encoder_gguf ? encoder_gguf : "";
    std::string dec_path = decoder_gguf ? decoder_gguf : "";
    if (enc_path.empty()) {
        enc_path = aicore_querysplat_model_default("encoder");
        if (enc_path.empty()) return nullptr;
    }
    if (dec_path.empty()) {
        dec_path = aicore_querysplat_model_default("decoder");
        if (dec_path.empty()) return nullptr;
    }

    auto* ctx = new (std::nothrow) aicore_querysplat_ctx();
    if (!ctx) return nullptr;
    ctx->encoder_path = enc_path;
    ctx->decoder_path = dec_path;
    ctx->device_request = o.common.device;
    ctx->threads = o.common.threads;
    ctx->f32_accuracy = o.f32_accuracy;
    ctx->image_height = o.image_height;
    ctx->image_width = o.image_width;
    std::memcpy(ctx->bg_color, o.bg_color, sizeof(ctx->bg_color));
    ctx->tto_lr = o.tto_lr;
    ctx->tto_lpips_weight = o.tto_lpips_weight;
    ctx->opacity_threshold = o.opacity_threshold;
    ctx->log_level = o.log_level;
    ctx->dump_dir = o.dump_dir;

    map_anything::set_log_level(
            (map_anything::LogLevel)std::min(3, std::max(0, o.log_level)));

    // decoder metadata first: the model resolution + gaussian layout are
    // needed before anything else (the official options.yaml reads)
    if (!qs::read_decoder_config(dec_path, &ctx->meta_cfg)) {
        ctx->last_error = "failed to read the decoder GGUF metadata";
        return ctx;  // not ready; last_error carries the reason
    }

    // Vulkan f32-accuracy instance profile: applied through the shared env
    // bridge BEFORE the backend exists (ggml snapshots instance knobs at
    // creation). CUDA/CPU parity runs at graph level (exact attention +
    // dequant casts); Vulkan's flash is F32-exact only with these knobs.
    if (o.f32_accuracy) {
        const std::string dev = o.common.device;
        const bool vulkan =
                dev.rfind("vulkan", 0) == 0 || dev == "auto" || dev.empty();
        if (vulkan) {
            aicore::apply_vulkan_f32_accuracy_profile();
        }
    }

    map_anything::EncoderRuntime::Config ecfg;
    ecfg.encoder_gguf = enc_path;
    ecfg.device = o.common.device;
    ecfg.n_threads = o.common.threads;
    ecfg.f32_accuracy = o.f32_accuracy != 0;
    ecfg.dump_dir = o.dump_dir;
    {
        std::string err;
        if (!ctx->enc.load(ecfg, &err)) {
            ctx->last_error = err.empty() ? "encoder load failed (see the "
                                            "[MapAnything] log lines)"
                                          : err;
            return ctx;  // not ready
        }
    }
    ctx->resolved_device = ctx->enc.backend().name;

    qs::DecoderRuntimeOptions dopts;
    dopts.f32_accuracy = o.f32_accuracy != 0;
    dopts.dump_dir = o.dump_dir;
    ctx->dec.reset(
            new qs::QuerySplatDecoder(dec_path, ctx->enc.backend(), dopts));
    if (!ctx->dec->valid()) {
        ctx->last_error = "failed to load the decoder GGUF (see the log)";
        return ctx;  // not ready
    }
    ctx->last_error.clear();
    return ctx;
}

AICORE_CAPI void aicore_querysplat_free(aicore_querysplat_ctx* ctx) {
    delete ctx;
}

AICORE_CAPI int aicore_querysplat_is_ready(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->enc.valid() && ctx->dec && ctx->dec->valid() ? 1 : 0;
}

AICORE_CAPI const char* aicore_querysplat_last_error(
        const aicore_querysplat_ctx* ctx) {
    return ctx && !ctx->last_error.empty() ? ctx->last_error.c_str() : nullptr;
}

AICORE_CAPI void aicore_querysplat_free_buffer(void* p) { free(p); }

AICORE_CAPI int32_t
aicore_querysplat_default_image_size(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->dec && ctx->dec->valid()
                   ? (int32_t)ctx->meta_cfg.img_size[0]
                   : 0;
}

AICORE_CAPI int aicore_querysplat_run_images(aicore_querysplat_ctx* ctx,
                                             const aicore_image_view* views,
                                             int32_t n_views) {
    ScopedTaskLogTag taskLogTag;
    if (!ctx || !views || n_views < 1) {
        if (ctx) ctx->last_error = "invalid arguments (ctx/views/n_views)";
        return -1;
    }
    if (!aicore_querysplat_is_ready(ctx)) {
        ctx->last_error = "context is not ready";
        return -1;
    }
    // resolved resolution: explicit options override, else decoder metadata
    const int h = ctx->image_height > 0 ? ctx->image_height
                                        : (int)ctx->meta_cfg.img_size[0];
    const int w = ctx->image_width > 0 ? ctx->image_width
                                       : (int)ctx->meta_cfg.img_size[1];
    const double t0 = steady_ms();
    std::vector<float> frames;
    frames.reserve((size_t)n_views * 3 * h * w);
    for (int32_t i = 0; i < n_views; i++) {
        const aicore_image_view& v = views[i];
        const int channels = v.format == AICORE_IMAGE_GRAY8   ? 1
                             : v.format == AICORE_IMAGE_RGB8  ? 3
                             : v.format == AICORE_IMAGE_BGR8  ? 3
                             : v.format == AICORE_IMAGE_RGBA8 ? 4
                             : v.format == AICORE_IMAGE_BGRA8 ? 4
                                                              : 0;
        if (channels == 0 || !v.data || v.width <= 0 || v.height <= 0 ||
            v.row_stride_bytes < (size_t)v.width * channels) {
            ctx->last_error = "invalid image view (format/dimensions/stride)";
            return -1;
        }
        std::vector<float> planes;
        if (!qs::preprocess_view_bytes(v.data, v.width, v.height,
                                       int(v.row_stride_bytes), channels,
                                       v.format == AICORE_IMAGE_BGR8 ||
                                               v.format == AICORE_IMAGE_BGRA8,
                                       h, w, &planes)) {
            ctx->last_error = "image preprocess failed";
            return -1;
        }
        frames.insert(frames.end(), planes.begin(), planes.end());
    }
    const double pp = steady_ms() - t0;
    return run_common(ctx, std::move(frames), (int)n_views, h, w, pp, t0);
}

AICORE_CAPI int aicore_querysplat_run_frames(aicore_querysplat_ctx* ctx,
                                             const float* frames_f32,
                                             int32_t n_views,
                                             int32_t height,
                                             int32_t width) {
    ScopedTaskLogTag taskLogTag;
    if (!ctx || !frames_f32 || n_views < 1 || height <= 0 || width <= 0) {
        if (ctx) ctx->last_error = "invalid arguments (ctx/frames/n/h/w)";
        return -1;
    }
    if (!aicore_querysplat_is_ready(ctx)) {
        ctx->last_error = "context is not ready";
        return -1;
    }
    const double t0 = steady_ms();
    const size_t expect = (size_t)n_views * 3 * height * width;
    std::vector<float> input(frames_f32, frames_f32 + expect);
    const double pp = steady_ms() - t0;
    return run_common(ctx, std::move(input), (int)n_views, (int)height,
                      (int)width, pp, t0);
}

// ---- result accessors ----

AICORE_CAPI int32_t
aicore_querysplat_result_view_count(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->r_views : 0;
}

AICORE_CAPI int32_t
aicore_querysplat_result_height(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->r_height : 0;
}

AICORE_CAPI int32_t
aicore_querysplat_result_width(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->r_width : 0;
}

AICORE_CAPI const float* aicore_querysplat_result_pose(
        const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->r_pose_store.data() : nullptr;
}

AICORE_CAPI const float* aicore_querysplat_result_w2c(
        const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->r_cams.w2c.data() : nullptr;
}

AICORE_CAPI const float* aicore_querysplat_result_cam2world(
        const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->r_cams.c2w.data() : nullptr;
}

AICORE_CAPI const float* aicore_querysplat_result_intrinsics(
        const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->r_cams.intrinsics.data() : nullptr;
}

AICORE_CAPI const float* aicore_querysplat_result_depth(
        const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result && !ctx->enc_out.depth.empty()
                   ? ctx->enc_out.depth.data()
                   : nullptr;
}

AICORE_CAPI const float* aicore_querysplat_result_depth_conf(
        const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result && !ctx->enc_out.depth_conf.empty()
                   ? ctx->enc_out.depth_conf.data()
                   : nullptr;
}

AICORE_CAPI int64_t
aicore_querysplat_result_gaussian_count(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->dec_out.N : 0;
}

AICORE_CAPI int32_t
aicore_querysplat_result_gaussian_stride(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? (11 + ctx->dec_out.color_dim) : 0;
}

AICORE_CAPI const float* aicore_querysplat_result_gaussians(
        const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->has_result ? ctx->dec_out.gaussians.data() : nullptr;
}

AICORE_CAPI int32_t
aicore_querysplat_result_sh_degree(const aicore_querysplat_ctx* ctx) {
    return ctx && ctx->dec && ctx->dec->valid()
                   ? (int32_t)ctx->meta_cfg.gaussian_sh_degree
                   : 0;
}

AICORE_CAPI int aicore_querysplat_last_pipeline_timings(
        const aicore_querysplat_ctx* ctx, aicore_pipeline_timings* out) {
    if (!ctx || !out) return -1;
    if (!ctx->has_pipeline) return -1;
    *out = ctx->pipeline;
    return 0;
}

// ---- rendering (CPU rasterizer, gsplat semantics) ----

AICORE_CAPI int aicore_querysplat_render_views(aicore_querysplat_ctx* ctx,
                                               const float* w2c,
                                               const float* intrinsics,
                                               int32_t n_views,
                                               const float** out_rgb,
                                               unsigned char** out_rgb8) {
    ScopedTaskLogTag taskLogTag;
    if (!ctx || !w2c || !intrinsics || n_views < 1 || !out_rgb || !out_rgb8) {
        if (ctx) ctx->last_error = "invalid arguments (render)";
        return -1;
    }
    if (!ctx->has_result) {
        ctx->last_error = "no result to render (run first)";
        return -2;
    }
    const qs::DecoderConfig& cfg = ctx->meta_cfg;
    qs::RenderConfig rcfg;
    rcfg.width = ctx->r_width;
    rcfg.height = ctx->r_height;
    rcfg.near_plane = cfg.znear;
    rcfg.far_plane = cfg.zfar;
    rcfg.sh_degree = cfg.gaussian_sh_degree;
    std::memcpy(rcfg.bg_color, ctx->bg_color, sizeof(rcfg.bg_color));

    qs::GaussianOutput g;
    g.N = ctx->dec_out.N;
    g.color_dim = ctx->dec_out.color_dim;
    g.data = ctx->dec_out.gaussians;  // borrowed copy into the output struct
    g.opacity.resize((size_t)g.N);
    const int stride = 11 + g.color_dim;
    for (int64_t i = 0; i < g.N; i++)
        g.opacity[(size_t)i] = g.data[(size_t)i * stride + 3];
    if (aicore_cancel_requested()) {
        ctx->last_error = "cancelled";
        return -2;
    }
    ctx->render_out =
            qs::render_gaussians(g, w2c, intrinsics, (int)n_views, rcfg);
    const size_t plane = (size_t)ctx->render_out.width * ctx->render_out.height;
    ctx->render_rgb8.assign(ctx->render_out.images.size(), 0);
    for (int32_t v = 0; v < n_views; v++) {
        qs::render_to_rgb8(
                ctx->render_out.images.data() + (size_t)v * 3 * plane,
                ctx->render_out.width, ctx->render_out.height,
                ctx->render_rgb8.data() + (size_t)v * plane * 3);
    }
    *out_rgb = ctx->render_out.images.data();
    *out_rgb8 = ctx->render_rgb8.data();
    ctx->last_error.clear();
    return 0;
}

// ---- TTO ----

AICORE_CAPI int aicore_querysplat_run_tto(
        aicore_querysplat_ctx* ctx,
        int32_t n_steps,
        aicore_querysplat_tto_step** out_losses,
        int32_t* out_n_losses,
        const float** out_gaussians,
        int64_t* out_count) {
    ScopedTaskLogTag taskLogTag;
    if (!ctx || n_steps < 1 || !out_losses || !out_n_losses || !out_gaussians ||
        !out_count) {
        if (ctx) ctx->last_error = "invalid arguments (run_tto)";
        return -1;
    }
    if (!ctx->has_result) {
        ctx->last_error = "no result to optimize (run first)";
        return -2;
    }
    if (ctx->meta_cfg.gaussian_sh_degree != 0) {
        ctx->last_error =
                "TTO supports sh0 checkpoints only (the official chain)";
        return -1;
    }
    auto lease_lock = ctx->enc.backend().lease.lock();
    // LPIPS metric graph (t3) for the weighted loss term; resolved from the
    // shared cache root exactly like the decoder/encoder weights.
    std::unique_ptr<qs::LpipsVgg> lpips;
    if (ctx->tto_lpips_weight > 0) {
        std::string lp =
                aicore::querysplat_model_cache_dir() + "/lpips-vgg-f32.gguf";
        lpips.reset(new qs::LpipsVgg(lp, ctx->enc.backend()));
        if (!lpips->valid()) {
            ctx->last_error = "missing LPIPS metric weights " + lp +
                              " (resolve through the model catalog)";
            return -2;
        }
    }
    const int rc = run_tto_loop(ctx, (int)n_steps, lpips);
    if (rc != 0) return rc;
    ctx->tto_gaussians_store = ctx->dec_out.gaussians;
    *out_losses = ctx->tto_losses.data();
    *out_n_losses = (int32_t)ctx->tto_losses.size();
    *out_gaussians = ctx->tto_gaussians_store.data();
    *out_count = ctx->dec_out.N;
    ctx->last_error.clear();
    return 0;
}

// ---- exports ----

AICORE_CAPI int aicore_querysplat_export_gaussians_ply(
        const aicore_querysplat_ctx* ctx,
        const char* path,
        float opacity_threshold) {
    if (!ctx || !path || !*path) return -1;
    if (!ctx->has_result) return -2;
    qs::GaussianOutput g;
    g.N = ctx->dec_out.N;
    g.color_dim = ctx->dec_out.color_dim;
    g.data = ctx->dec_out.gaussians;
    g.opacity.resize((size_t)g.N);
    const int stride = 11 + g.color_dim;
    for (int64_t i = 0; i < g.N; i++)
        g.opacity[(size_t)i] = g.data[(size_t)i * stride + 3];
    return qs::save_gaussian_ply(g, path, opacity_threshold) ? 0 : -2;
}

AICORE_CAPI char* aicore_querysplat_export_cameras_json(
        const aicore_querysplat_ctx* ctx) {
    if (!ctx || !ctx->has_result) return nullptr;
    std::ostringstream json;
    json << "{\"n_views\":" << ctx->r_views << ",\"height\":" << ctx->r_height
         << ",\"width\":" << ctx->r_width << ",\"pose_stride\":9,\"views\":[";
    for (int s = 0; s < ctx->r_views; s++) {
        if (s) json << ",";
        json << "{\"view\":" << s << ",\"pose_enc\":[";
        for (int k = 0; k < 9; k++)
            json << (k ? "," : "") << ctx->r_pose_store[(size_t)s * 9 + k];
        json << "],\"c2w\":[";
        for (int k = 0; k < 16; k++)
            json << (k ? "," : "") << ctx->r_cams.c2w[(size_t)s * 16 + k];
        json << "],\"intrinsics\":[";
        for (int k = 0; k < 4; k++)
            json << (k ? "," : "") << ctx->r_cams.intrinsics[(size_t)s * 4 + k];
        json << "]}";
    }
    json << "]}";
    return dup_cstr(json.str());
}

AICORE_CAPI int aicore_querysplat_depth_pointcloud(
        const aicore_querysplat_ctx* ctx,
        int32_t target_points,
        const float** out_xyz,
        unsigned char** out_rgb,
        int32_t* out_n) {
    if (!ctx || target_points <= 0 || !out_xyz || !out_rgb || !out_n) return -1;
    if (!ctx->has_result || ctx->enc_out.depth.empty()) return -2;
    // save_vggt_depth_pointcloud semantics: confidence-proportional sampling
    // (depth_conf >= 1; weight = conf - 1), unprojection through the
    // predicted cameras, RGB from the normalized input frames.
    const int S = ctx->r_views, H = ctx->r_height, W = ctx->r_width;
    const size_t plane = (size_t)H * W;
    std::vector<double> weight(plane * S);
    double wsum = 0.0;
    for (size_t i = 0; i < weight.size(); i++) {
        weight[i] = std::max(0.0, (double)ctx->enc_out.depth_conf[i] - 1.0);
        wsum += weight[i];
    }
    if (wsum <= 0.0) {
        // degenerate confidence: uniform sampling
        std::fill(weight.begin(), weight.end(), 1.0);
        wsum = (double)weight.size();
    }
    aicore_querysplat_ctx* nc = const_cast<aicore_querysplat_ctx*>(ctx);
    nc->pc_xyz.clear();
    nc->pc_rgb.clear();
    nc->pc_xyz.reserve((size_t)target_points * 3);
    nc->pc_rgb.reserve((size_t)target_points * 3);
    // systematic stride sampling over the cumulative weight (deterministic)
    double acc = 0.0;
    const double step_w = wsum / (double)target_points;
    size_t next = 0;
    for (double t = step_w * 0.5; t < wsum && next < weight.size();) {
        while (next < weight.size() && acc + weight[next] < t) {
            acc += weight[next];
            next++;
        }
        if (next >= weight.size()) break;
        const int s = (int)(next / plane);
        const size_t p = next % plane;
        const int y = (int)(p / W), x = (int)(p % W);
        const float d = ctx->enc_out.depth[next];
        // camera-frame ray: ((x+0.5-cx)/fx, (y+0.5-cy)/fy, 1) * d
        const float fx = ctx->r_cams.intrinsics[(size_t)s * 4 + 0];
        const float fy = ctx->r_cams.intrinsics[(size_t)s * 4 + 1];
        const float cx = ctx->r_cams.intrinsics[(size_t)s * 4 + 2];
        const float cy = ctx->r_cams.intrinsics[(size_t)s * 4 + 3];
        float cam[3] = {((float)x + 0.5f - cx) / fx * d,
                        ((float)y + 0.5f - cy) / fy * d, d};
        // c2w (row-major 4x4) * cam
        const float* m = &ctx->r_cams.c2w[(size_t)s * 16];
        for (int r = 0; r < 3; r++)
            nc->pc_xyz.push_back(m[r * 4 + 0] * cam[0] + m[r * 4 + 1] * cam[1] +
                                 m[r * 4 + 2] * cam[2] + m[r * 4 + 3]);
        // [0,1] input rgb -> u8 (the observed sRGB pixels, NOT the
        // ImageNet-normalized planes)
        for (int c = 0; c < 3; c++) {
            float v = ctx->r_frames_01[(size_t)s * 3 * plane +
                                       (size_t)c * plane + p];
            v = std::min(std::max(v, 0.0f), 1.0f);
            nc->pc_rgb.push_back((unsigned char)(v * 255.0f));
        }
        t += step_w;
    }
    *out_xyz = nc->pc_xyz.data();
    *out_rgb = nc->pc_rgb.data();
    *out_n = (int32_t)(nc->pc_xyz.size() / 3);
    return 0;
}

AICORE_CAPI char* aicore_querysplat_export_distribution_stats(
        const aicore_querysplat_ctx* ctx) {
    if (!ctx || !ctx->has_result) return nullptr;
    // save_gaussian_alpha_distribution / scale_distribution: the histogram
    // data behind the official plots (32 bins over [0,1] / log-scale span).
    const int64_t N = ctx->dec_out.N;
    const int stride = 11 + ctx->dec_out.color_dim;
    const float* g = ctx->dec_out.gaussians.data();
    constexpr int kBins = 32;
    double amin = 1.0, amax = 0.0, asum = 0.0;
    double smin = 1e30, smax = -1e30, ssum_log = 0.0;
    int64_t abins[kBins] = {0};
    int64_t sbins[kBins] = {0};
    for (int64_t i = 0; i < N; i++) {
        const float o = g[i * stride + 3];
        amin = std::min(amin, (double)o);
        amax = std::max(amax, (double)o);
        asum += o;
        {
            int b = (int)(o * kBins);
            b = std::min(kBins - 1, std::max(0, b));
            abins[b]++;
        }
        const float* sc = &g[i * stride + 4];
        for (int c = 0; c < 3; c++) {
            const double lv = std::log((double)sc[c] + 1e-8);
            smin = std::min(smin, lv);
            smax = std::max(smax, lv);
            ssum_log += lv;
        }
    }
    std::ostringstream json;
    json << "{\"n_gaussians\":" << N;
    json << ",\"opacity\":{\"min\":" << amin << ",\"max\":" << amax
         << ",\"mean\":" << (N ? asum / (double)N : 0.0) << ",\"bins\":[";
    for (int b = 0; b < kBins; b++) json << (b ? "," : "") << abins[b];
    json << "]}";
    if (N) {
        const double smean = ssum_log / (3.0 * (double)N);
        json << ",\"scale_log\":{\"min\":" << smin << ",\"max\":" << smax
             << ",\"mean\":" << smean << ",\"bins\":[";
        for (int64_t i = 0; i < N; i++) {
            const float* sc = &g[i * stride + 4];
            for (int c = 0; c < 3; c++) {
                const double lv = std::log((double)sc[c] + 1e-8);
                int b = (int)(((lv - smin) / (smax - smin + 1e-30)) * kBins);
                b = std::min(kBins - 1, std::max(0, b));
                sbins[b]++;
            }
        }
        for (int b = 0; b < kBins; b++) json << (b ? "," : "") << sbins[b];
        json << "]}";
    }
    json << "}";
    return dup_cstr(json.str());
}

AICORE_CAPI char* aicore_querysplat_info_json(
        const aicore_querysplat_ctx* ctx) {
    if (ctx == nullptr) return nullptr;
    std::ostringstream json;
    json << "{\"task\":\"querysplat\"";
    json << ",\"encoder\":\"" << json_escape(ctx->encoder_path) << "\"";
    json << ",\"decoder\":\"" << json_escape(ctx->decoder_path) << "\"";
    json << ",\"device\":\"" << json_escape(ctx->resolved_device) << "\"";
    json << ",\"device_request\":\"" << json_escape(ctx->device_request)
         << "\"";
    json << ",\"threads\":" << ctx->threads;
    json << ",\"f32_accuracy\":" << (ctx->f32_accuracy ? "true" : "false");
    json << ",\"image_size\":["
         << (ctx->image_height > 0 ? ctx->image_height
                                   : (int)ctx->meta_cfg.img_size[0])
         << ","
         << (ctx->image_width > 0 ? ctx->image_width
                                  : (int)ctx->meta_cfg.img_size[1])
         << "]";
    json << ",\"default_image_size\":"
         << aicore_querysplat_default_image_size(ctx);
    json << ",\"sh_degree\":" << ctx->meta_cfg.gaussian_sh_degree;
    json << ",\"bg_color\":[" << ctx->bg_color[0] << "," << ctx->bg_color[1]
         << "," << ctx->bg_color[2] << "]";
    json << ",\"tto\":{\"lr\":" << ctx->tto_lr
         << ",\"lpips_weight\":" << ctx->tto_lpips_weight << "}";
    json << ",\"opacity_threshold\":" << ctx->opacity_threshold;
    json << ",\"ready\":"
         << (aicore_querysplat_is_ready(ctx) ? "true" : "false");
    json << ",\"result\":";
    if (ctx->has_result) {
        json << "{\"views\":" << ctx->r_views;
        json << ",\"height\":" << ctx->r_height;
        json << ",\"width\":" << ctx->r_width;
        json << ",\"n_gaussians\":" << ctx->dec_out.N;
        json << ",\"gaussian_stride\":" << (11 + ctx->dec_out.color_dim);
        json << ",\"e2e_ms\":" << ctx->pipeline.e2e_ms;
        json << "}";
    } else {
        json << "null";
    }
    json << "}";
    return dup_cstr(json.str());
}

// ---- model catalog ----

#include "aicore/asset_digests.h"

namespace {

// Download base mirrors the map_anything catalog convention (HF resolve
// main; ecvModelDownloader follows the 302 automatically).
constexpr const char* kQuerySplatDownloadBase =
        "https://huggingface.co/Asher-1/querysplat_vggto/resolve/main/";

struct QuerySplatModelRow {
    const char* filename;
    const char* display_name;
    const char* role;  // "decoder" | "lpips"
    const char* quant_note;
    const char* license_note;
    const char* variant;
    int64_t size_bytes;
};

constexpr QuerySplatModelRow kQuerySplatModels[] = {
        {"querysplat-paper-q8_0.gguf", "QuerySplat paper decoder", "decoder",
         "Q8_0 - smallest, same quality vs GT",
         "QuerySplat Apache-2.0; VGGT-Omega lineage: verify the upstream "
         "VGGT license before commercial use",
         "paper", 1307495456},
        {"querysplat-paper-f16.gguf", "QuerySplat paper decoder", "decoder",
         "F16 - half size, weights staged as F32 at load",
         "QuerySplat Apache-2.0; VGGT-Omega lineage: verify the upstream "
         "VGGT license before commercial use",
         "paper", 2302209056},
        {"querysplat-paper-f32.gguf", "QuerySplat paper decoder", "decoder",
         "F32 - parity reference",
         "QuerySplat Apache-2.0; VGGT-Omega lineage: verify the upstream "
         "VGGT license before commercial use",
         "paper", 4467002400},
        {"lpips-vgg-f32.gguf", "LPIPS-VGG metric (TTO)", "lpips",
         "F32 - metric graph", "LPIPS VGG: see the upstream license notes",
         "lpips", 58866656},
};

constexpr int kQuerySplatModelCount =
        (int)(sizeof(kQuerySplatModels) / sizeof(kQuerySplatModels[0]));
constexpr int kQuerySplatDefaultDecoder = 0;  // q8_0: smallest fast default

const aicore_querysplat_model_entry* querysplat_catalog_rows() {
    static const std::vector<aicore_querysplat_model_entry> rows = [] {
        std::vector<aicore_querysplat_model_entry> v;
        v.reserve(kQuerySplatModelCount);
        for (const auto& m : kQuerySplatModels) {
            aicore_querysplat_model_entry e{};
            e.filename = m.filename;
            e.display_name = m.display_name;
            e.role = m.role;
            e.quant_note = m.quant_note;
            e.license_note = m.license_note;
            e.variant = m.variant;
            e.size_bytes = m.size_bytes;
            static std::vector<std::string> url_store;
            url_store.push_back(std::string(kQuerySplatDownloadBase) +
                                m.filename);
            e.download_url = url_store.back().c_str();
            v.push_back(e);
        }
        return v;
    }();
    return rows.data();
}

}  // namespace

AICORE_CAPI int32_t aicore_querysplat_model_count(void) {
    return kQuerySplatModelCount;
}

AICORE_CAPI const aicore_querysplat_model_entry* aicore_querysplat_model_at(
        int32_t index) {
    if (index < 0 || index >= kQuerySplatModelCount) return nullptr;
    return querysplat_catalog_rows() + index;
}

AICORE_CAPI const char* aicore_querysplat_model_default(const char* role) {
    if (role == nullptr) return nullptr;
    static std::string encoder_default;
    if (std::strcmp(role, "encoder") == 0) {
        // delegate to the map_anything catalog: the VGGT-Omega 1B/512 q8_0
        // file is the shared encoder (single source of truth)
        const int idx = aicore_map_anything_model_default_index();
        const auto* e = aicore_map_anything_model_at(idx);
        if (!e) return nullptr;
        static std::string path;
        path = aicore::map_anything_model_cache_dir() + "/" + e->filename;
        return path.c_str();
    }
    if (std::strcmp(role, "decoder") == 0) {
        static std::string path;
        path = aicore::querysplat_model_cache_dir() + "/" +
               kQuerySplatModels[kQuerySplatDefaultDecoder].filename;
        return path.c_str();
    }
    if (std::strcmp(role, "lpips") == 0) {
        static std::string path;
        path = aicore::querysplat_model_cache_dir() + "/lpips-vgg-f32.gguf";
        return path.c_str();
    }
    return nullptr;
}

AICORE_CAPI char* aicore_querysplat_model_resolve_cache_path(
        const char* filename) {
    if (!filename || !*filename) return nullptr;
    for (int i = 0; i < kQuerySplatModelCount; ++i) {
        if (std::strcmp(kQuerySplatModels[i].filename, filename) == 0) {
            return dup_cstr(aicore::querysplat_model_cache_dir() + "/" +
                            filename);
        }
    }
    // encoder files resolve through the map_anything cache root
    if (aicore::AssetDigestForFile(filename) != nullptr &&
        std::strstr(filename, "vggt-omega") == filename) {
        return dup_cstr(aicore::map_anything_model_cache_dir() + "/" +
                        filename);
    }
    return nullptr;
}
