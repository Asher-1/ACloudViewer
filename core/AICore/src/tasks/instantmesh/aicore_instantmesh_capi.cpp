// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------
//
// Public C ABI for the InstantMesh image-to-3D task. Wraps
// instantmesh_pipeline.hpp; no Qt, STL, exceptions, or ggml types cross this
// boundary (image decode uses QImage internally, like trellis).

#include <QImage>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#include "aicore/backend_capi.h"
#include "aicore/instantmesh_capi.h"
#include "aicore/runtime_capi.h"
#include "common/aicore_log.hpp"
#include "common/capi_utils.hpp"
#include "common/model_cache.hpp"
#include "instantmesh_pipeline.hpp"

using instantmesh::GenParams;
using instantmesh::PipelineConfig;
using instantmesh::PipelineContext;
using instantmesh::PipelineMesh;

namespace {

struct Options {
    std::string device = "auto";
    int threads = 0;
    std::string rmbg_gguf;
    int steps = 75;
    uint64_t seed = 42;
    float guidance = 4.0f;
    int grid_res = 64;
    float grid_scale = 2.1f;
    int texture_res = 2048;
    int bake_texture = 1;
};

struct Ctx {
    PipelineContext pipe;
    Options opts;  // load-time copy (defaults for generate params)
    std::string backend;
    std::string error;
    aicore_pipeline_timings last_timings{};
    bool has_timings = false;
};

// Shared by both public input paths: finalize timing last to include
// destruction of decode/packing buffers inside try.
struct ApiTimingsFinalizer {
    Ctx *ctx;
    std::chrono::steady_clock::time_point start;
    ~ApiTimingsFinalizer() noexcept {
        if (!ctx->has_timings) return;
        auto &timings = ctx->last_timings;
        const double elapsed = std::chrono::duration<double, std::milli>(
                                       std::chrono::steady_clock::now() - start)
                                       .count();
        timings.preprocess_ms += elapsed - timings.e2e_ms;
        timings.e2e_ms = elapsed;
    }
};

void copy_err(char *err, int err_len, const char *msg) noexcept {
    if (err && err_len > 0)
        std::snprintf(err, static_cast<size_t>(err_len), "%s", msg);
}

void copy_err(char *err, int err_len, const std::string &msg) noexcept {
    copy_err(err, err_len, msg.c_str());
}

template <typename T>
bool valid_struct(const T *value, char *err, int err_len) noexcept {
    if (value->struct_size < sizeof(T)) {
        copy_err(err, err_len, "invalid struct_size");
        return false;
    }
    if (value->abi_version != AICORE_INSTANTMESH_ABI_VERSION) {
        copy_err(err, err_len, "unsupported abi_version");
        return false;
    }
    return true;
}

bool valid_numeric(const aicore_instantmesh_generate_params &p,
                   bool allow_defaults,
                   char *err,
                   int err_len) noexcept {
    const auto omitted = [allow_defaults](double value) {
        return allow_defaults && value <= 0;
    };
    const char *bad = nullptr;
    if (!omitted(p.steps) && (p.steps < 2 || p.steps > 1000))
        bad = "steps";
    else if (!std::isfinite(p.guidance) ||
             (!omitted(p.guidance) && (p.guidance <= 0 || p.guidance > 15)))
        bad = "guidance";
    else if (!omitted(p.grid_res) && (p.grid_res < 2 || p.grid_res > 256))
        bad = "grid_res";
    else if (!std::isfinite(p.grid_scale) ||
             (!omitted(p.grid_scale) && p.grid_scale <= 0))
        bad = "grid_scale";
    else if (!omitted(p.texture_res) &&
             (p.texture_res < 1 || p.texture_res > 4096))
        bad = "texture_res";
    else if (p.bake_texture != 0 && p.bake_texture != 1 &&
             !(allow_defaults && p.bake_texture == -1))
        bad = "bake_texture";
    if (bad) copy_err(err, err_len, bad);
    return bad == nullptr;
}

// Task-private format boundary shared by generation and model-free
// preprocessing; preserve input alpha.
bool pack_rgba(const aicore_image_view *view,
               std::vector<uint8_t> &rgba,
               std::string &error) {
    if (!view || !view->data || view->width <= 0 || view->height <= 0 ||
        static_cast<uint64_t>(view->width) * view->height > 16u * 1024 * 1024) {
        error = "invalid image dimensions or data";
        return false;
    }
    size_t channels = 0;
    switch (view->format) {
        case AICORE_IMAGE_GRAY8:
            channels = 1;
            break;
        case AICORE_IMAGE_RGB8:
        case AICORE_IMAGE_BGR8:
            channels = 3;
            break;
        case AICORE_IMAGE_RGBA8:
        case AICORE_IMAGE_BGRA8:
            channels = 4;
            break;
        default:
            error = "unsupported image format";
            return false;
    }
    const size_t row = static_cast<size_t>(view->width) * channels;
    const size_t max_span =
            static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
    if (view->row_stride_bytes < row || row > max_span ||
        (view->height > 1 &&
         view->row_stride_bytes >
                 (max_span - row) / static_cast<size_t>(view->height - 1))) {
        error = "invalid image stride or span overflow";
        return false;
    }
    rgba.resize(static_cast<size_t>(view->width) * view->height * 4);
    const bool bgr = view->format == AICORE_IMAGE_BGR8 ||
                     view->format == AICORE_IMAGE_BGRA8;
    for (int y = 0; y < view->height; ++y) {
        const auto *src =
                view->data + static_cast<size_t>(y) * view->row_stride_bytes;
        auto *dst = rgba.data() + static_cast<size_t>(y) * view->width * 4;
        for (int x = 0; x < view->width; ++x, src += channels, dst += 4) {
            dst[0] = src[channels == 1 ? 0 : (bgr ? 2 : 0)];
            dst[1] = src[channels == 1 ? 0 : 1];
            dst[2] = src[channels == 1 ? 0 : (bgr ? 0 : 2)];
            dst[3] = channels == 4 ? src[3] : 255;
        }
    }
    return true;
}

bool decode_image_rgba(const void *image_bytes,
                       int image_len,
                       std::vector<uint8_t> &rgba,
                       int &w,
                       int &h,
                       std::string &e) {
    if (!image_bytes || image_len <= 0) {
        e = "invalid image bytes";
        return false;
    }
    QImage img = QImage::fromData((const uchar *)image_bytes, image_len);
    if (img.isNull()) {
        e = "image decode failed (unsupported format?)";
        return false;
    }
    if (img.width() <= 0 || img.height() <= 0 ||
        (int64_t)img.width() * img.height() > (int64_t)16 * 1024 * 1024) {
        e = "image dimensions out of range";
        return false;
    }
    const QImage conv = img.convertToFormat(QImage::Format_RGBA8888);
    w = conv.width();
    h = conv.height();
    rgba.resize((size_t)w * h * 4);
    std::memcpy(rgba.data(), conv.constBits(), rgba.size());
    return true;
}

GenParams to_gen_params(const aicore_instantmesh_generate_params *params,
                        const Options &opts) {
    GenParams p;
    p.seed = opts.seed;
    if (params) {
        p.steps = params->steps;
        p.seed = params->seed;
        p.guidance = params->guidance;
        p.grid_res = params->grid_res;
        p.grid_scale = params->grid_scale;
        p.texture_res = params->texture_res;
        p.bake_texture = params->bake_texture;
    }
    if (p.steps <= 0) p.steps = opts.steps;
    if (p.guidance <= 0) p.guidance = opts.guidance;
    if (p.grid_res <= 0) p.grid_res = opts.grid_res;
    if (p.grid_scale <= 0) p.grid_scale = opts.grid_scale;
    if (p.texture_res <= 0) p.texture_res = opts.texture_res;
    if (p.bake_texture < 0) p.bake_texture = opts.bake_texture;
    return p;
}

aicore_instantmesh_mesh *generation_failure(Ctx *c,
                                            const char *message,
                                            char *err,
                                            int err_len) noexcept {
    if (c) {
        c->has_timings = false;
        try {
            c->error = message;
        } catch (...) {
        }
    }
    copy_err(err, err_len, message);
    return nullptr;
}

Ctx *begin_generate(aicore_instantmesh_ctx *ctx,
                    const aicore_instantmesh_generate_params *params,
                    char *err,
                    int err_len) noexcept {
    auto *c = reinterpret_cast<Ctx *>(ctx);
    if (c) {
        c->has_timings = false;
        c->error.clear();
    }
    copy_err(err, err_len, "");
    char validation_error[128] = {};
    if (params &&
        (!valid_struct(params, validation_error, sizeof(validation_error)) ||
         !valid_numeric(*params, true, validation_error,
                        sizeof(validation_error)))) {
        generation_failure(c, validation_error, err, err_len);
        return nullptr;
    }
    if (!c) copy_err(err, err_len, "null context");
    return c;
}

aicore_instantmesh_mesh *generate_rgba(
        Ctx *c,
        const std::vector<uint8_t> &rgba,
        int w,
        int h,
        const aicore_instantmesh_generate_params *params,
        aicore_instantmesh_progress_fn progress,
        void *user,
        char *err,
        int err_len,
        std::chrono::steady_clock::time_point api_start) {
    const auto pipeline_start = std::chrono::steady_clock::now();
    PipelineMesh mesh;
    aicore_pipeline_timings timings{};
    std::string error;
    const GenParams gp = to_gen_params(params, c->opts);
    if (!c->pipe.generate(rgba.data(), w, h, w * 4, gp, progress, user, &mesh,
                          &timings, &error))
        return generation_failure(c, error.c_str(), err, err_len);
    const auto native_start = std::chrono::steady_clock::now();
    auto result = std::make_unique<PipelineMesh>(std::move(mesh));
    const auto native_ready = std::chrono::steady_clock::now();
    // The C ABI is the public input boundary; preprocessing includes
    // validation, decoding and stride/format packing.
    timings.preprocess_ms += std::chrono::duration<double, std::milli>(
                                     pipeline_start - api_start)
                                     .count();
    timings.postprocess_ms += std::chrono::duration<double, std::milli>(
                                      native_ready - native_start)
                                      .count();
    timings.e2e_ms =
            std::chrono::duration<double, std::milli>(native_ready - api_start)
                    .count();
    timings.valid_fields |= AICORE_TIMING_PREPROCESS |
                            AICORE_TIMING_POSTPROCESS | AICORE_TIMING_E2E;
    c->last_timings = timings;
    c->has_timings = true;
    return reinterpret_cast<aicore_instantmesh_mesh *>(result.release());
}

}  // namespace

extern "C" {

int aicore_instantmesh_abi_version(void) {
    return AICORE_INSTANTMESH_ABI_VERSION;
}

aicore_instantmesh_options *aicore_instantmesh_options_new(void) {
    return reinterpret_cast<aicore_instantmesh_options *>(new (std::nothrow)
                                                                  Options());
}

void aicore_instantmesh_options_free(aicore_instantmesh_options *opts) {
    delete reinterpret_cast<Options *>(opts);
}

void aicore_instantmesh_options_set_device(aicore_instantmesh_options *opts,
                                           const char *device) {
    if (opts)
        reinterpret_cast<Options *>(opts)->device =
                device && device[0] ? device : "auto";
}

void aicore_instantmesh_options_set_threads(aicore_instantmesh_options *opts,
                                            int n_threads) {
    if (opts) reinterpret_cast<Options *>(opts)->threads = n_threads;
}

void aicore_instantmesh_options_set_rmbg_gguf(aicore_instantmesh_options *opts,
                                              const char *rmbg_gguf) {
    if (opts)
        reinterpret_cast<Options *>(opts)->rmbg_gguf =
                rmbg_gguf ? rmbg_gguf : "";
}

void aicore_instantmesh_options_set_steps(aicore_instantmesh_options *opts,
                                          int steps) {
    if (opts) reinterpret_cast<Options *>(opts)->steps = steps;
}

void aicore_instantmesh_options_set_seed(aicore_instantmesh_options *opts,
                                         uint64_t seed) {
    if (opts) reinterpret_cast<Options *>(opts)->seed = seed;
}

void aicore_instantmesh_options_set_guidance(aicore_instantmesh_options *opts,
                                             float guidance) {
    if (opts) reinterpret_cast<Options *>(opts)->guidance = guidance;
}

void aicore_instantmesh_options_set_grid_res(aicore_instantmesh_options *opts,
                                             int grid_res) {
    if (opts) reinterpret_cast<Options *>(opts)->grid_res = grid_res;
}

void aicore_instantmesh_options_set_grid_scale(aicore_instantmesh_options *opts,
                                               float grid_scale) {
    if (opts) reinterpret_cast<Options *>(opts)->grid_scale = grid_scale;
}

void aicore_instantmesh_options_set_texture_res(
        aicore_instantmesh_options *opts, int texture_res) {
    if (opts) reinterpret_cast<Options *>(opts)->texture_res = texture_res;
}

void aicore_instantmesh_options_set_bake_texture(
        aicore_instantmesh_options *opts, int bake_texture) {
    if (opts) reinterpret_cast<Options *>(opts)->bake_texture = bake_texture;
}

aicore_instantmesh_ctx *aicore_instantmesh_load_opts(
        const aicore_instantmesh_model_paths *paths,
        const aicore_instantmesh_options *opts) {
    char err[512] = {};
    auto *ctx = aicore_instantmesh_load_opts_ex(paths, opts, err, sizeof(err));
    if (!ctx) AICORE_LOG_ERROR("[instantmesh] ", "load failed: %s\n", err);
    return ctx;
}

aicore_instantmesh_ctx *aicore_instantmesh_load_opts_ex(
        const aicore_instantmesh_model_paths *paths,
        const aicore_instantmesh_options *opts,
        char *err,
        int err_len) {
    copy_err(err, err_len, "");
    if (!paths) {
        copy_err(err, err_len, "null model paths");
        return nullptr;
    }
    if (!valid_struct(paths, err, err_len)) return nullptr;
    try {
        const auto supplied = [](const char *path) { return path && path[0]; };
        const int n_diffusion = supplied(paths->cond_gguf) +
                                supplied(paths->unet_gguf) +
                                supplied(paths->vae_gguf);
        if (n_diffusion != 0 && n_diffusion != 3) {
            copy_err(err, err_len, "incomplete Zero123++ model trio");
            return nullptr;
        }
        if (!supplied(paths->dino_gguf) || !supplied(paths->transformer_gguf) ||
            !supplied(paths->synthesizer_gguf)) {
            copy_err(err, err_len, "required geometry model paths missing");
            return nullptr;
        }
        auto ctx = std::make_unique<Ctx>();
        if (opts) ctx->opts = *reinterpret_cast<const Options *>(opts);
        aicore_instantmesh_generate_params defaults =
                AICORE_INSTANTMESH_GENERATE_PARAMS_INIT;
        defaults.steps = ctx->opts.steps;
        defaults.guidance = ctx->opts.guidance;
        defaults.grid_res = ctx->opts.grid_res;
        defaults.grid_scale = ctx->opts.grid_scale;
        defaults.texture_res = ctx->opts.texture_res;
        defaults.bake_texture = ctx->opts.bake_texture;
        if (!valid_numeric(defaults, false, err, err_len)) return nullptr;
        PipelineConfig cfg;
        cfg.cond_gguf = paths->cond_gguf ? paths->cond_gguf : "";
        cfg.unet_gguf = paths->unet_gguf ? paths->unet_gguf : "";
        cfg.vae_gguf = paths->vae_gguf ? paths->vae_gguf : "";
        cfg.dino_gguf = paths->dino_gguf ? paths->dino_gguf : "";
        cfg.transformer_gguf =
                paths->transformer_gguf ? paths->transformer_gguf : "";
        cfg.synthesizer_gguf =
                paths->synthesizer_gguf ? paths->synthesizer_gguf : "";
        cfg.rmbg_gguf = ctx->opts.rmbg_gguf;
        cfg.device = ctx->opts.device;
        cfg.threads = ctx->opts.threads;
        cfg.steps = ctx->opts.steps;
        cfg.seed = ctx->opts.seed;
        cfg.guidance = ctx->opts.guidance;
        cfg.grid_res = ctx->opts.grid_res;
        cfg.grid_scale = ctx->opts.grid_scale;
        cfg.texture_res = ctx->opts.texture_res;
        cfg.bake_texture = ctx->opts.bake_texture != 0;

        std::string error;
        if (!ctx->pipe.load(cfg, &error)) {
            AICORE_LOG_ERROR("[instantmesh] ", "load failed: %s\n",
                             error.c_str());
            copy_err(err, err_len, error);
            return nullptr;
        }
        ctx->backend = ctx->pipe.backend_name();
        return reinterpret_cast<aicore_instantmesh_ctx *>(ctx.release());
    } catch (const std::exception &e) {
        copy_err(err, err_len, e.what());
    } catch (...) {
        copy_err(err, err_len, "unknown load exception");
    }
    return nullptr;
}

void aicore_instantmesh_free(aicore_instantmesh_ctx *ctx) {
    delete reinterpret_cast<Ctx *>(ctx);
}

int aicore_instantmesh_is_ready(const aicore_instantmesh_ctx *ctx) {
    return ctx && reinterpret_cast<const Ctx *>(ctx)->pipe.is_ready() ? 1 : 0;
}

const char *aicore_instantmesh_last_error(const aicore_instantmesh_ctx *ctx) {
    return ctx ? reinterpret_cast<const Ctx *>(ctx)->error.c_str() : "";
}

int aicore_instantmesh_has_multiview(const aicore_instantmesh_ctx *ctx) {
    return ctx && reinterpret_cast<const Ctx *>(ctx)->pipe.has_multiview() ? 1
                                                                           : 0;
}

const char *aicore_instantmesh_backend(const aicore_instantmesh_ctx *ctx) {
    return ctx ? reinterpret_cast<const Ctx *>(ctx)->backend.c_str() : "";
}

void aicore_instantmesh_free_buffer(void *p) { std::free(p); }

aicore_instantmesh_mesh *aicore_instantmesh_generate(
        aicore_instantmesh_ctx *ctx,
        const void *image_bytes,
        int image_len,
        const aicore_instantmesh_generate_params *params,
        aicore_instantmesh_progress_fn progress,
        void *progress_user,
        char *err,
        int err_len) {
    const auto api_start = std::chrono::steady_clock::now();
    auto *c = begin_generate(ctx, params, err, err_len);
    if (!c) return nullptr;
    const ApiTimingsFinalizer finalize{c, api_start};
    try {
        std::vector<uint8_t> rgba;
        int w = 0, h = 0;
        std::string error;
        if (!decode_image_rgba(image_bytes, image_len, rgba, w, h, error))
            return generation_failure(c, error.c_str(), err, err_len);
        return generate_rgba(c, rgba, w, h, params, progress, progress_user,
                             err, err_len, api_start);
    } catch (const std::exception &e) {
        return generation_failure(c, e.what(), err, err_len);
    } catch (...) {
        return generation_failure(c, "unknown generate exception", err,
                                  err_len);
    }
}

aicore_instantmesh_mesh *aicore_instantmesh_generate_image_view(
        aicore_instantmesh_ctx *ctx,
        const aicore_image_view *image,
        const aicore_instantmesh_generate_params *params,
        aicore_instantmesh_progress_fn progress,
        void *progress_user,
        char *err,
        int err_len) {
    const auto api_start = std::chrono::steady_clock::now();
    auto *c = begin_generate(ctx, params, err, err_len);
    if (!c) return nullptr;
    const ApiTimingsFinalizer finalize{c, api_start};
    try {
        std::vector<uint8_t> rgba;
        std::string error;
        if (!pack_rgba(image, rgba, error))
            return generation_failure(c, error.c_str(), err, err_len);
        return generate_rgba(c, rgba, image->width, image->height, params,
                             progress, progress_user, err, err_len, api_start);
    } catch (const std::exception &e) {
        return generation_failure(c, e.what(), err, err_len);
    } catch (...) {
        return generation_failure(c, "unknown generate exception", err,
                                  err_len);
    }
}

int aicore_instantmesh_last_pipeline_timings(const aicore_instantmesh_ctx *ctx,
                                             aicore_pipeline_timings *out) {
    auto *c = const_cast<Ctx *>(reinterpret_cast<const Ctx *>(ctx));
    if (!c || !out || !c->has_timings) return -1;
    *out = c->last_timings;
    return 0;
}

int aicore_instantmesh_mesh_n_verts(const aicore_instantmesh_mesh *r) {
    return r ? (int)(reinterpret_cast<const PipelineMesh *>(r)->verts.size() /
                     3)
             : 0;
}

int aicore_instantmesh_mesh_n_tris(const aicore_instantmesh_mesh *r) {
    return r ? (int)(reinterpret_cast<const PipelineMesh *>(r)->faces.size() /
                     3)
             : 0;
}

const float *aicore_instantmesh_mesh_verts(const aicore_instantmesh_mesh *r) {
    return r ? reinterpret_cast<const PipelineMesh *>(r)->verts.data()
             : nullptr;
}

const int *aicore_instantmesh_mesh_tris(const aicore_instantmesh_mesh *r) {
    return r ? reinterpret_cast<const PipelineMesh *>(r)->faces.data()
             : nullptr;
}

const float *aicore_instantmesh_mesh_vcolors(const aicore_instantmesh_mesh *r) {
    return r ? reinterpret_cast<const PipelineMesh *>(r)->vcolors.data()
             : nullptr;
}

const float *aicore_instantmesh_mesh_uvs(const aicore_instantmesh_mesh *r) {
    return r ? reinterpret_cast<const PipelineMesh *>(r)->uvs.data() : nullptr;
}

const uint8_t *aicore_instantmesh_mesh_texture(
        const aicore_instantmesh_mesh *r) {
    return r ? reinterpret_cast<const PipelineMesh *>(r)->tex_rgba.data()
             : nullptr;
}

int aicore_instantmesh_mesh_texture_w(const aicore_instantmesh_mesh *r) {
    return r ? reinterpret_cast<const PipelineMesh *>(r)->tex_w : 0;
}

int aicore_instantmesh_mesh_texture_h(const aicore_instantmesh_mesh *r) {
    return r ? reinterpret_cast<const PipelineMesh *>(r)->tex_h : 0;
}

void aicore_instantmesh_mesh_free(aicore_instantmesh_mesh *r) {
    delete reinterpret_cast<PipelineMesh *>(r);
}

uint8_t *aicore_instantmesh_bake_glb(const float *verts,
                                     int n_verts,
                                     const int *tris,
                                     int n_tris,
                                     const float *uvs,
                                     const uint8_t *texture_rgba,
                                     int tex_w,
                                     int tex_h,
                                     int *out_len,
                                     char *err,
                                     int err_len) {
    if (out_len) *out_len = 0;
    copy_err(err, err_len, "");
    try {
        std::string error;
        uint8_t *glb = instantmesh::bake_glb_buffer(
                verts, n_verts, tris, n_tris, uvs, texture_rgba, tex_w, tex_h,
                out_len, &error);
        if (!glb) copy_err(err, err_len, error);
        return glb;
    } catch (const std::exception &e) {
        copy_err(err, err_len, e.what());
    } catch (...) {
        copy_err(err, err_len, "unknown GLB bake exception");
    }
    return nullptr;
}

int aicore_instantmesh_preprocess_image_bytes(const void *image_bytes,
                                              int image_len,
                                              int out_size,
                                              unsigned char *out_rgb,
                                              char *err,
                                              int err_len) {
    if (!image_bytes || image_len <= 0 || out_size <= 0 || out_size > 4096 ||
        !out_rgb) {
        copy_err(err, err_len, "invalid arguments");
        return 1;
    }
    try {
        std::vector<uint8_t> rgba;
        int w = 0, h = 0;
        std::string derr;
        if (!decode_image_rgba(image_bytes, image_len, rgba, w, h, derr)) {
            copy_err(err, err_len, derr);
            return 1;
        }
        const aicore_image_view view{rgba.data(), w, h,
                                     static_cast<size_t>(w) * 4,
                                     AICORE_IMAGE_RGBA8};
        return aicore_instantmesh_preprocess_image_view(&view, out_size,
                                                        out_rgb, err, err_len);
    } catch (...) {
        copy_err(err, err_len, "decode exception");
        return 1;
    }
}

int aicore_instantmesh_preprocess_image_view(const aicore_image_view *image,
                                             int out_size,
                                             unsigned char *out_rgb,
                                             char *err,
                                             int err_len) {
    copy_err(err, err_len, "");
    if (!out_rgb || out_size < 1 || out_size > 4096) {
        copy_err(err, err_len, "invalid output dimensions or buffer");
        return 1;
    }
    try {
        std::vector<uint8_t> rgba;
        std::string error;
        if (!pack_rgba(image, rgba, error)) {
            copy_err(err, err_len, error);
            return 1;
        }
        const int w = image->width, h = image->height;
        // Alpha compositing over white retains the original rounding semantics.
        std::vector<uint8_t> rgb((size_t)w * h * 3);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const uint8_t *px = &rgba[((size_t)y * w + x) * 4];
                const float a = px[3] / 255.f;
                for (int c = 0; c < 3; ++c)
                    rgb[((size_t)y * w + x) * 3 + c] =
                            (uint8_t)(px[c] * a + 255.f * (1.f - a));
            }
        if (!instantmesh::preprocess_rgb(rgb.data(), w, h, out_size, out_rgb)) {
            copy_err(err, err_len, "preprocess failed");
            return 1;
        }
        return 0;
    } catch (...) {
        copy_err(err, err_len, "preprocess exception");
        return 1;
    }
}

int aicore_instantmesh_warmup_backend(const char *device) {
    try {
        return aicore_warmup_backend(device != nullptr ? device : "auto");
    } catch (...) {
        return -1;
    }
}

void aicore_instantmesh_shutdown(void) { aicore_runtime_shutdown(); }

char *aicore_instantmesh_model_cache_dir(void) {
    try {
        return aicore::capi::dup_cstr(aicore::instantmesh_model_cache_dir());
    } catch (...) {
        return nullptr;
    }
}

char *aicore_instantmesh_info_json(aicore_instantmesh_ctx *ctx) {
    if (!ctx) return aicore::capi::dup_cstr("{\"error\":\"null context\"}");
    try {
        auto *c = reinterpret_cast<Ctx *>(ctx);
        std::string j = "{";
        j += "\"multiview\":" +
             std::string(c->pipe.has_multiview() ? "true" : "false");
        j += ",\"backend\":\"" + aicore::capi::json_escape(c->backend) + "\"";
        j += "}";
        return aicore::capi::dup_cstr(j);
    } catch (...) {
        return nullptr;
    }
}

}  // extern "C"
