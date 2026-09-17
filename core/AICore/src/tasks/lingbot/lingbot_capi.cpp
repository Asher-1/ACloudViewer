// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// LingBot-Map C ABI implementation. Wraps the in-tree cpp_ggml engine port
// (lingbot_model / lingbot_skyseg) with opaque contexts, explicit options,
// typed borrowed results, the common timing contract, and the published
// Hugging Face model catalog (Asher-1/lingbot-map-gguf).

#include "aicore/lingbot_capi.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/runtime_capi.h"
#include "capi_utils.hpp"
#include "common/data_root_util.hpp"
#include "common/ggml_backend_utils.hpp"
#include "common/model_cache.hpp"
#include "lingbot_model.h"
#include "lingbot_skyseg.h"

namespace {

constexpr int kAbiVersion = 1;

struct Options {
    std::string device;
    int threads = 0;
    int image_size = 518;
    int kv_scale = 8;
    int kv_window = 64;
    int stream_capacity = 0;
};

struct Ctx {
    Options opt;
    lingbot::model model;
    lingbot::skyseg sky;
    bool sky_ready = false;

    // Typed-result storage backing the borrowed aicore_lingbot_result.
    std::vector<float> depth, conf, c2w, intr, pose_enc;
    // Per-frame sky keep-mask (255 = keep, 0 = sky), at the processed
    // resolution; populated by the native skyseg pass or by an external
    // set_external_sky_masks injection (one-shot, consumed by the next
    // infer_stream).
    std::vector<std::vector<unsigned char>> sky_masks;
    std::vector<std::vector<unsigned char>> external_masks;
    int32_t external_mask_w = 0;
    int32_t external_mask_h = 0;
    aicore_lingbot_result last{};
    int stream_frames = 0;
    aicore_pipeline_timings timings{};

    std::string error;
};

Options* as_opts(aicore_lingbot_options* p) {
    return reinterpret_cast<Options*>(p);
}
const Options* as_opts(const aicore_lingbot_options* p) {
    return reinterpret_cast<const Options*>(p);
}
Ctx* as_ctx(aicore_lingbot_ctx* p) { return reinterpret_cast<Ctx*>(p); }
const Ctx* as_ctx(const aicore_lingbot_ctx* p) {
    return reinterpret_cast<const Ctx*>(p);
}

// ---------------------------------------------------------------------------
// Official preprocessing (mode="crop" of the upstream
// load_and_preprocess_images): aspect-preserving bicubic resize to
// image_size width, height snapped to the patch grid, center crop when it
// exceeds image_size, output as normalized [0,1] NCHW RGB. ImageNet
// normalization happens inside the engine graph.
// ---------------------------------------------------------------------------

}  // namespace

#include "pillow_resample.inc"

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

AICORE_CAPI aicore_lingbot_options* aicore_lingbot_options_new(void) {
    return reinterpret_cast<aicore_lingbot_options*>(new Options());
}

AICORE_CAPI void aicore_lingbot_options_free(aicore_lingbot_options* opts) {
    delete as_opts(opts);
}

AICORE_CAPI void aicore_lingbot_options_set_device(aicore_lingbot_options* opts,
                                                   const char* device) {
    if (Options* o = as_opts(opts)) o->device = device ? device : "";
}

AICORE_CAPI void aicore_lingbot_options_set_threads(
        aicore_lingbot_options* opts, int n_threads) {
    if (Options* o = as_opts(opts)) o->threads = n_threads;
}

AICORE_CAPI void aicore_lingbot_options_set_image_size(
        aicore_lingbot_options* opts, int size) {
    if (Options* o = as_opts(opts)) o->image_size = size;
}

AICORE_CAPI void aicore_lingbot_options_set_kv_profile(
        aicore_lingbot_options* opts, int scale_frames, int window_frames) {
    if (Options* o = as_opts(opts)) {
        o->kv_scale = scale_frames;
        o->kv_window = window_frames;
    }
}

AICORE_CAPI void aicore_lingbot_options_set_stream_capacity(
        aicore_lingbot_options* opts, int total_frames) {
    if (Options* o = as_opts(opts)) o->stream_capacity = total_frames;
}

// ---------------------------------------------------------------------------
// Context lifecycle
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_lingbot_abi_version(void) { return kAbiVersion; }

AICORE_CAPI aicore_lingbot_ctx* aicore_lingbot_load_opts(
        const char* gguf_path, const aicore_lingbot_options* opts) {
    if (!gguf_path || !*gguf_path) return nullptr;
    Ctx* ctx = new Ctx();
    if (opts) ctx->opt = *as_opts(opts);

    lingbot::model_options mo;
    // Official release profile: persistent F16 KV cache, scale=8/window=64.
    mo.kv_cache_scale = ctx->opt.kv_scale;
    mo.kv_cache_window = ctx->opt.kv_window;
    mo.kv_f16 = lingbot::kv_f16_mode::strict;
    if (ctx->opt.stream_capacity > 0)
        mo.kv_total_frames = ctx->opt.stream_capacity;

    if (!ctx->model.load(gguf_path,
                         ctx->opt.device.empty() ? "auto" : ctx->opt.device,
                         ctx->opt.threads > 0 ? ctx->opt.threads : 4, mo)) {
        // AICore contract: a context is either ready or carries a queryable
        // error — return the non-ready context instead of NULL so callers
        // can read aicore_lingbot_last_error for the reason.
        ctx->error = ctx->model.error();
    }
    return reinterpret_cast<aicore_lingbot_ctx*>(ctx);
}

AICORE_CAPI void aicore_lingbot_free(aicore_lingbot_ctx* ctx) {
    delete as_ctx(ctx);
}

AICORE_CAPI int aicore_lingbot_is_ready(const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    return (c && !c->model.device_name().empty() &&
            c->model.info().image_size > 0)
                   ? 1
                   : 0;
}

AICORE_CAPI const char* aicore_lingbot_last_error(
        const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    return c ? c->error.c_str() : "";
}

AICORE_CAPI void aicore_lingbot_free_buffer(void* p) { std::free(p); }

// ---------------------------------------------------------------------------
// Preprocessing
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_lingbot_preprocess_image(const aicore_image_view* image,
                                                int image_size,
                                                float* out_nchw,
                                                int out_size,
                                                int32_t* out_width,
                                                int32_t* out_height) {
    if (!image || !image->data || image->width <= 0 || image->height <= 0 ||
        image_size <= 0 || image->format < AICORE_IMAGE_RGB8 ||
        image->format > AICORE_IMAGE_BGRA8) {
        return -1;
    }
    const int w0 = image->width;
    const int h0 = image->height;
    const int patch = 14;
    // Official crop: new_width = image_size, height snapped to the patch grid
    // (round to a multiple of patch_size, minimum one patch).
    const int new_w = image_size;
    int snapped_h = static_cast<int>(
            std::lround(static_cast<double>(h0) * new_w / w0 / patch) * patch);
    if (snapped_h < patch) snapped_h = patch;

    // Final output size AFTER the center crop: the sizing call and the
    // data call must agree on the same NCHW element count.
    const int out_h_final = snapped_h > image_size ? image_size : snapped_h;
    const size_t needed = static_cast<size_t>(new_w) * out_h_final * 3;
    if (!out_nchw) return static_cast<int>(needed);
    if (out_size < static_cast<int>(needed)) return -1;

    // 1) Decode the borrowed view into u8 interleaved RGB (stride-aware).
    std::vector<unsigned char> rgb(static_cast<size_t>(h0) * w0 * 3);
    for (int y = 0; y < h0; ++y) {
        const uint8_t* row =
                image->data + static_cast<size_t>(y) * image->row_stride_bytes;
        unsigned char* dst = rgb.data() + static_cast<size_t>(y) * w0 * 3;
        switch (image->format) {
            case AICORE_IMAGE_RGB8:
                std::memcpy(dst, row, static_cast<size_t>(w0) * 3);
                break;
            case AICORE_IMAGE_RGBA8:
                for (int x = 0; x < w0; ++x) {
                    dst[x * 3 + 0] = row[x * 4 + 0];
                    dst[x * 3 + 1] = row[x * 4 + 1];
                    dst[x * 3 + 2] = row[x * 4 + 2];
                }
                break;
            case AICORE_IMAGE_BGR8:
                for (int x = 0; x < w0; ++x) {
                    dst[x * 3 + 0] = row[x * 3 + 2];
                    dst[x * 3 + 1] = row[x * 3 + 1];
                    dst[x * 3 + 2] = row[x * 3 + 0];
                }
                break;
            case AICORE_IMAGE_BGRA8:
                for (int x = 0; x < w0; ++x) {
                    dst[x * 3 + 0] = row[x * 4 + 2];
                    dst[x * 3 + 1] = row[x * 4 + 1];
                    dst[x * 3 + 2] = row[x * 4 + 0];
                }
                break;
            case AICORE_IMAGE_GRAY8:
                for (int x = 0; x < w0; ++x) {
                    dst[x * 3 + 0] = dst[x * 3 + 1] = dst[x * 3 + 2] = row[x];
                }
                break;
            default:
                return -1;
        }
    }

    // 2) Bicubic resize with the exact Pillow 12.2.0 8bpc pipeline
    //    (two-pass horizontal/vertical, fixed-point coefficients, u8
    //    intermediate). Identity axes are skipped exactly like Pillow's
    //    need_horizontal/need_vertical, so already-processed frames (the
    //    official scene datasets at 518-wide) are bit-identical no-ops.
    std::vector<uint8_t> stage;
    const uint8_t* stageData = rgb.data();
    int stageW = w0, stageH = h0;
    if (new_w != w0) {
        stage.resize(static_cast<size_t>(new_w) * h0 * 3);
        pillow_resample_horizontal(rgb.data(), w0, h0, stage.data(), new_w,
                                   pillow_precompute_coeffs(w0, new_w));
        stageData = stage.data();
        stageW = new_w;
    }
    if (snapped_h != h0) {
        std::vector<uint8_t> vertical(static_cast<size_t>(stageW) * snapped_h *
                                      3);
        pillow_resample_vertical(stageData, stageW, h0, vertical.data(),
                                 snapped_h,
                                 pillow_precompute_coeffs(h0, snapped_h));
        stageData = vertical.data();
        stageH = snapped_h;
        stage.swap(vertical);
        if (new_w != w0) {
            // horizontal already ran; keep the latest buffer alive
        }
    }
    const std::vector<uint8_t> resizedOwned(
            stageData, stageData + static_cast<size_t>(stageW) * stageH * 3);
    const uint8_t* resized = resizedOwned.data();

    // 3) Center crop when the snapped height exceeds image_size (official
    //    crop rule: resize keeps the aspect, then crop vertically centered).
    int out_h = snapped_h;
    int crop_y0 = 0;
    if (snapped_h > image_size) {
        crop_y0 = (snapped_h - image_size) / 2;
        out_h = image_size;
    }

    // 4) HWC u8 -> NCHW [0,1] over the cropped region.
    for (int c = 0; c < 3; ++c) {
        float* plane = out_nchw + static_cast<size_t>(c) * out_h * new_w;
        for (int y = 0; y < out_h; ++y) {
            const uint8_t* src =
                    resized + (static_cast<size_t>(y + crop_y0) * new_w) * 3;
            for (int x = 0; x < new_w; ++x) {
                plane[static_cast<size_t>(y) * new_w + x] =
                        src[static_cast<size_t>(x) * 3 + c] / 255.0f;
            }
        }
    }
    if (out_width) *out_width = new_w;
    if (out_height) *out_height = out_h;
    return static_cast<int>(needed);
}

// ---------------------------------------------------------------------------
// Streaming inference
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_lingbot_infer_stream(aicore_lingbot_ctx* ctx,
                                            const float* frames_nchw,
                                            int n_frames,
                                            int width,
                                            int height,
                                            aicore_lingbot_frame_cb cb,
                                            void* user) {
    Ctx* c = as_ctx(ctx);
    if (!c || !frames_nchw || n_frames <= 0 || width <= 0 || height <= 0) {
        if (c) c->error = "invalid stream arguments";
        return -1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    c->error.clear();
    c->stream_frames = 0;
    c->timings = {};

    // Optional sky masking, ahead of inference. An externally injected
    // mask set (official <scene>_sky_masks cache) takes precedence and
    // skips the native skyseg pass entirely; otherwise a loaded skyseg
    // model segments every frame first. Either way the sky depth
    // confidence is zeroed per frame during result delivery.
    c->sky_masks.clear();
    const bool external_masks = !c->external_masks.empty() &&
                                c->external_mask_w == width &&
                                c->external_mask_h == height;
    if (external_masks) {
        c->sky_masks = std::move(c->external_masks);
        c->external_masks.clear();
        c->external_mask_w = 0;
        c->external_mask_h = 0;
    } else if (c->sky_ready) {
        c->sky_masks.resize(static_cast<size_t>(n_frames));
        std::vector<float> hwc(static_cast<size_t>(width) * height * 3);
        std::vector<float> sky_map;
        for (int f = 0; f < n_frames; ++f) {
            const float* chw =
                    frames_nchw + static_cast<size_t>(f) * 3 * width * height;
            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    for (int ch = 0; ch < 3; ++ch) {
                        hwc[(static_cast<size_t>(y) * width + x) * 3 + ch] =
                                chw[(static_cast<size_t>(ch) * height + y) *
                                            width +
                                    x];
                    }
                }
            }
            if (!c->sky.infer(hwc.data(), height, width, sky_map)) {
                c->error = c->sky.error();
                return -1;
            }
            // Official postprocess (segment_sky_from_array): min-max
            // normalize, truncate to u8, resize, keep only pixels whose
            // resized u8 value is 0.
            float mn = sky_map[0], mx = sky_map[0];
            for (float v : sky_map) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            const float denom = std::max(mx - mn, 1e-8f);
            // NOTE: never name a local `small` — rpcndr.h (via windows.h)
            // #defines it to `char`, which breaks MSVC compilation.
            std::vector<unsigned char> sky_u8(320 * 320);
            for (size_t i = 0; i < sky_u8.size(); ++i) {
                sky_u8[i] = static_cast<unsigned char>((sky_map[i] - mn) *
                                                       255.0f / denom);
            }
            std::vector<unsigned char>& keep = c->sky_masks[f];
            keep.assign(static_cast<size_t>(width) * height, 0);
            std::vector<unsigned char> u8map(static_cast<size_t>(width) *
                                             height);
            lingbot::skyseg_resize_mask_u8(sky_u8.data(), 320, 320,
                                           u8map.data(), width, height);
            for (size_t i = 0; i < u8map.size(); ++i)
                keep[i] = (u8map[i] == 0) ? 255 : 0;
        }
    }

    // Wrap the C callback into the engine's std::function contract. The
    // borrowed result points into ctx-owned storage that stays stable until
    // the next stream call or free.
    auto engine_cb = [&](int frame_index, const lingbot::output& out) -> bool {
        c->depth = out.depth;
        c->conf = out.depth_conf;
        c->c2w = out.c2w;
        c->intr = out.intrinsics;
        c->pose_enc = out.pose_enc;
        c->last.depth = c->depth.data();
        c->last.depth_conf = c->conf.data();
        c->last.c2w = c->c2w.data();
        c->last.intrinsics = c->intr.data();
        c->last.pose_enc = c->pose_enc.data();
        c->last.width = width;
        c->last.height = height;
        c->last.frame_index = frame_index;

        if (!c->sky_masks.empty() && frame_index >= 0 &&
            frame_index < static_cast<int>(c->sky_masks.size())) {
            const std::vector<unsigned char>& keep = c->sky_masks[frame_index];
            if (keep.size() == c->conf.size()) {
                // Official --mask_sky semantics: zero the depth confidence
                // of sky pixels (keep == 0).
                for (size_t i = 0; i < c->conf.size(); ++i) {
                    if (keep[i] == 0) c->conf[i] = 0.0f;
                }
                c->last.depth_conf = c->conf.data();
            }
        }
        // C ABI convention: 0 = continue, non-zero = abort. The engine's
        // std::function uses bool (true = continue) — invert explicitly.
        if (cb && cb(user, &c->last) != 0) {
            c->error = "frame callback aborted";
            return false;
        }
        ++c->stream_frames;
        return true;
    };
    c->model.set_frame_callback(engine_cb);

    lingbot::output result;
    if (!c->model.infer(frames_nchw, 1, n_frames, 3, height, width, &result)) {
        c->error = c->model.error();
        return -1;
    }
    aicore::capi::record_pipeline_e2e(c->timings, t0);
    return 0;
}

AICORE_CAPI int aicore_lingbot_stream_reset(aicore_lingbot_ctx* ctx) {
    Ctx* c = as_ctx(ctx);
    if (!c) return -1;
    c->model.reset_cache();
    return 0;
}

AICORE_CAPI int aicore_lingbot_last_stream_frames(
        const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    return c ? c->stream_frames : -1;
}

AICORE_CAPI int aicore_lingbot_last_pipeline_timings(
        const aicore_lingbot_ctx* ctx, aicore_pipeline_timings* out) {
    const Ctx* c = as_ctx(ctx);
    if (!c || !out) return -1;
    *out = c->timings;
    return 0;
}

// ---------------------------------------------------------------------------
// Sky segmentation
// ---------------------------------------------------------------------------

AICORE_CAPI int aicore_lingbot_skyseg_load(aicore_lingbot_ctx* ctx,
                                           const char* skyseg_gguf_path) {
    Ctx* c = as_ctx(ctx);
    if (!c || !skyseg_gguf_path || !*skyseg_gguf_path) return -1;
    // The sky model shares the streaming model's backend lease (same device,
    // same execution queue, CLI-equivalent layout).
    if (!c->model.device_name().empty() &&
        c->sky.load(skyseg_gguf_path, c->model.backend_handle())) {
        c->sky_ready = true;
        return 0;
    }
    c->error = c->sky.error().empty() ? "skyseg load failed" : c->sky.error();
    return -1;
}

AICORE_CAPI int aicore_lingbot_skyseg_ready(const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    return (c && c->sky_ready) ? 1 : 0;
}

AICORE_CAPI int aicore_lingbot_set_external_sky_masks(
        aicore_lingbot_ctx* ctx,
        const unsigned char* masks,
        int frame_count,
        int width,
        int height) {
    Ctx* c = as_ctx(ctx);
    if (!c || !masks || frame_count <= 0 || width <= 0 || height <= 0) {
        return -1;
    }
    const size_t plane = static_cast<size_t>(width) * height;
    c->external_masks.assign(static_cast<size_t>(frame_count),
                             std::vector<unsigned char>());
    for (int f = 0; f < frame_count; ++f) {
        c->external_masks[static_cast<size_t>(f)].assign(
                masks + static_cast<size_t>(f) * plane,
                masks + static_cast<size_t>(f + 1) * plane);
    }
    c->external_mask_w = width;
    c->external_mask_h = height;
    return 0;
}

AICORE_CAPI int aicore_lingbot_last_sky_mask(aicore_lingbot_ctx* ctx,
                                             unsigned char* buf,
                                             int buf_size) {
    Ctx* c = as_ctx(ctx);
    if (!c || buf_size < 0) return -1;
    if (c->sky_masks.empty()) return c->sky_ready ? 0 : 0;
    // Most recent frame: the callback keeps last.frame_index current.
    const int idx = std::max(0, c->last.frame_index);
    if (idx >= static_cast<int>(c->sky_masks.size())) return 0;
    const std::vector<unsigned char>& keep = c->sky_masks[idx];
    const int needed = static_cast<int>(keep.size());
    if (!buf) return needed;
    if (buf_size < needed) return needed;
    std::memcpy(buf, keep.data(), static_cast<size_t>(needed));
    return needed;
}

// ---------------------------------------------------------------------------
// Introspection
// ---------------------------------------------------------------------------

AICORE_CAPI const char* aicore_lingbot_context_device(
        const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    static const std::string kEmpty;
    return c ? c->model.device_name().c_str() : kEmpty.c_str();
}

AICORE_CAPI int aicore_lingbot_context_threads(const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    return c ? c->model.threads() : 0;
}

AICORE_CAPI int aicore_lingbot_context_image_size(
        const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    return c ? c->model.info().image_size : 0;
}

AICORE_CAPI int aicore_lingbot_context_patch_size(
        const aicore_lingbot_ctx* ctx) {
    const Ctx* c = as_ctx(ctx);
    return c ? c->model.info().patch_size : 0;
}

AICORE_CAPI char* aicore_lingbot_info_json(aicore_lingbot_ctx* ctx) {
    Ctx* c = as_ctx(ctx);
    if (!c) return nullptr;
    const auto& info = c->model.info();
    std::string json = "{";
    json += "\"image_size\":" + std::to_string(info.image_size);
    json += ",\"patch_size\":" + std::to_string(info.patch_size);
    json += ",\"embed_dim\":" + std::to_string(info.embed_dim);
    json += ",\"block_count\":" + std::to_string(info.block_count);
    json += ",\"output_dim\":" + std::to_string(info.output_dim);
    json += ",\"weight_type\":\"" + info.weight_type + "\"";
    json += ",\"graph_version\":\"" + info.graph_version + "\"";
    json += ",\"device\":\"" + c->model.device_name() + "\"";
    json += ",\"skyseg\":" + std::string(c->sky_ready ? "true" : "false");
    json += "}";
    return aicore::capi::dup_cstr(json);
}

AICORE_CAPI int aicore_lingbot_warmup_backend(const char* device) {
    return aicore_warmup_backend(device != nullptr ? device : "auto");
}

AICORE_CAPI uint64_t aicore_lingbot_device_total_memory(const char* device) {
    if (!device || !*device) device = "auto";
    std::string family;
    int want_idx = 0;
    ggml_common::parse_device(device, family, want_idx);
    if (family == "cpu") return 0;
    ggml_common::load_backends_once();
    // "auto"/"gpu" resolve to the first accelerator of any family in runtime
    // order; an explicit family selects that backend's Nth device.
    std::string resolved;
    ggml_backend_t be = ggml_common::find_gpu_backend(
            family == "auto" ? "gpu" : family, want_idx, resolved);
    if (!be) return 0;
    ggml_backend_dev_t dev = ggml_backend_get_device(be);
    size_t free_bytes = 0, total_bytes = 0;
    if (dev) ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    ggml_backend_free(be);
    return static_cast<uint64_t>(total_bytes);
}

AICORE_CAPI void aicore_lingbot_shutdown(void) { aicore_runtime_shutdown(); }

AICORE_CAPI char* aicore_lingbot_model_cache_dir(void) {
    return aicore::capi::dup_cstr(aicore::lingbot_model_cache_dir());
}

// ---------------------------------------------------------------------------
// Published model catalog (Hugging Face Asher-1/lingbot-map-gguf)
// ---------------------------------------------------------------------------

namespace {

constexpr const char* kDownloadBase =
        "https://huggingface.co/Asher-1/lingbot-map-gguf/resolve/main/";
constexpr const char* kLicense = "Apache-2.0 (LingBot-Map; GGUF deployment)";

#define LINGBOT_ENTRY(file, display, quant, role, size, digest)            \
    {file,                                                                 \
     "https://huggingface.co/Asher-1/lingbot-map-gguf/resolve/main/" file, \
     display,                                                              \
     quant,                                                                \
     kLicense,                                                             \
     role,                                                                 \
     size##ULL,                                                            \
     digest}

const aicore_lingbot_model_entry kModels[] = {
        LINGBOT_ENTRY("lingbot-map-q8.gguf",
                      "LingBot-Map q8 (memory-saving)",
                      "Q8 8-bit quant",
                      "map",
                      1264126752,
                      "0a7273dce77cb7580a5fc2ca8019cd8f79ea725944e5790c0c2c4e61"
                      "7a7c2e36"),
        LINGBOT_ENTRY("lingbot-map-f16.gguf",
                      "LingBot-Map f16 (recommended)",
                      "F16 half precision",
                      "map",
                      2315989152,
                      "7f3d4816300efe0f3576ac7a0f8e16ea1891ef8f4986e749c193b93a"
                      "19e8acc5"),
        LINGBOT_ENTRY("lingbot-map-f32.gguf",
                      "LingBot-Map f32",
                      "F32 exact reference",
                      "map",
                      4631876256,
                      "0ccf3dd2cb649f0279c8d07b5c431921783283d3c7e175a5c06eaffa"
                      "2fa3b51c"),
        LINGBOT_ENTRY("lingbot-map-q4.gguf",
                      "LingBot-Map q4 (experimental)",
                      "Q4 4-bit quant (experimental)",
                      "map",
                      703133472,
                      "92e30cbbf367f4609de245889985490359aa08be29c3292d2fc8682e"
                      "e3875e52"),
        // Long-sequence checkpoints (upstream lingbot-map-long.pt): the
        // architecture is identical to the balanced ones (same tensor
        // names/shapes), so the graph and every option apply unchanged.
        LINGBOT_ENTRY("lingbot-map-long-q8.gguf",
                      "LingBot-Map long q8_0 (long sequences)",
                      "Q8 8-bit quant",
                      "map",
                      1264126752,
                      "729260e8b1639c6a22d100a1f188c213f0666bf074bfe777f3c25f98"
                      "5bf3805c"),
        LINGBOT_ENTRY("lingbot-map-long-f16.gguf",
                      "LingBot-Map long f16 (long sequences)",
                      "F16 half precision",
                      "map",
                      2315989152,
                      "309ef95b9883ff62cab7e34607d00281e4639387afda225416d934b6"
                      "2705a3a0"),
        LINGBOT_ENTRY("lingbot-map-long-f32.gguf",
                      "LingBot-Map long f32 (long sequences)",
                      "F32 exact reference",
                      "map",
                      4631876256,
                      "e6a13e8169125893f889e11888ffd40579af29c9c0ef20adbea7060a"
                      "5cb849a5"),
        LINGBOT_ENTRY("lingbot-map-skyseg-f16.gguf",
                      "SkySeg f16 (native sky mask)",
                      "F16 half precision",
                      "skyseg",
                      88011008,
                      "4e4ba738737a5f966950d24cc46b6fbdeb54c1598c081b01278b11f0"
                      "4d8fd211"),
        LINGBOT_ENTRY("lingbot-map-skyseg-q8_0.gguf",
                      "SkySeg q8_0 (native sky mask)",
                      "Q8 8-bit quant",
                      "skyseg",
                      46806944,
                      "759c2c0cc82e5493f48bfa7752c344c18bc13df5fa5fa36baaffe389"
                      "f66dcb47"),
        LINGBOT_ENTRY("lingbot-map-skyseg-f32.gguf",
                      "SkySeg f32 (native sky mask)",
                      "F32 exact reference",
                      "skyseg",
                      175946208,
                      "0409e16a891c59efbde1350e126241f83f3acc4b89c23fc5a9f8f98f"
                      "83feeb7f"),
};
constexpr int kModelCount =
        static_cast<int>(sizeof(kModels) / sizeof(kModels[0]));
constexpr int kDefaultMapIndex = 1;  // lingbot-map-f16.gguf — the upstream
// GUI default and full-alignment format (vs the official fp32 checkpoint:
// pose 1.72e-04 / depth 4.72e-04 over 286 frames, an order of magnitude
// tighter than the q8 quantization loss; validation_report.md). q8 stays the
// memory-saving deployment option for small-VRAM tiers.
// Sky-segmentation default inside the same 10-entry table (q8, f16, f32,
// q4, long-q8, long-f16, long-f32, skyseg-f16, skyseg-q8_0, skyseg-f32).
constexpr int kDefaultSkysegIndex = 7;  // lingbot-map-skyseg-f16.gguf

}  // namespace

AICORE_CAPI int aicore_lingbot_model_count(void) { return kModelCount; }

AICORE_CAPI const aicore_lingbot_model_entry* aicore_lingbot_model_at(
        int index) {
    return (index >= 0 && index < kModelCount) ? &kModels[index] : nullptr;
}

AICORE_CAPI int aicore_lingbot_model_default_index(void) {
    return kDefaultMapIndex;
}

AICORE_CAPI int aicore_lingbot_skyseg_model_default_index(void) {
    return kDefaultSkysegIndex;
}

AICORE_CAPI const aicore_lingbot_model_entry* aicore_lingbot_model_by_filename(
        const char* filename) {
    if (!filename) return nullptr;
    for (int i = 0; i < kModelCount; ++i) {
        if (std::strcmp(kModels[i].filename, filename) == 0) return &kModels[i];
    }
    return nullptr;
}

AICORE_CAPI const char* aicore_lingbot_model_download_base(void) {
    return kDownloadBase;
}
