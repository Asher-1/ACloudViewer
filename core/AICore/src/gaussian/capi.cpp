// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// C API implementation for FreeSplatterCore.
// Adapted from free-splatter.cpp/src/free_splatter.cpp to use the project's
// ggml_common, Qt image I/O, and PLY export modules.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <ggml_common/ggml_backend_utils.hpp>
#include <memory>
#include <string>
#include <vector>

#include "aicore/gaussian_capi.h"
#include "backend.hpp"
#include "image.h"
#include "image_io.hpp"
#include "model.h"
#include "options.h"
#include "path_util.hpp"
#include "ply_export.hpp"
#include "pose.h"

#if defined(GGML_USE_CUDA) && !defined(GGML_BACKEND_DL)
#include <cuda_runtime.h>
#endif

// ---- opaque types ----

struct aicore_gaussian_options {
    aicore::gaussian::options o;
};

struct aicore_gaussian_ctx {
    aicore::gaussian::model m;
    aicore::gaussian::options opts;
    std::string error;
};

// ---- ABI ----

extern "C" {

int AICORE_CAPI aicore_gaussian_abi_version(void) { return 1; }

// ---- options builder ----

aicore_gaussian_options* AICORE_CAPI aicore_gaussian_options_new(void) {
    return new aicore_gaussian_options();
}
void AICORE_CAPI aicore_gaussian_options_free(aicore_gaussian_options* opts) {
    delete opts;
}

void AICORE_CAPI aicore_gaussian_options_set_device(
        aicore_gaussian_options* opts, const char* device) {
    if (opts) opts->o.device = device ? device : "";
}
void AICORE_CAPI aicore_gaussian_options_set_threads(
        aicore_gaussian_options* opts, int n_threads) {
    if (opts) opts->o.n_threads = n_threads;
}
void AICORE_CAPI aicore_gaussian_options_set_dump_taps_dir(
        aicore_gaussian_options* opts, const char* dir) {
    if (opts) opts->o.dump_taps_dir = dir ? dir : "";
}

// ---- lifecycle ----

aicore_gaussian_ctx* AICORE_CAPI aicore_gaussian_load(const char* gguf_path,
                                                      int n_threads) {
    if (!gguf_path) return nullptr;
    auto* ctx = new (std::nothrow) aicore_gaussian_ctx();
    if (!ctx) return nullptr;
    ctx->opts.n_threads = n_threads;
    if (ctx->opts.device.empty()) ctx->opts.device = "auto";
    if (!ctx->m.load(gguf_path, ctx->opts.device, ctx->opts.n_threads)) {
        ctx->error = ctx->m.error;
    }
    return ctx;
}

aicore_gaussian_ctx* AICORE_CAPI aicore_gaussian_load_opts(
        const char* gguf_path, const aicore_gaussian_options* opts) {
    if (!gguf_path) return nullptr;
    auto* ctx = new (std::nothrow) aicore_gaussian_ctx();
    if (!ctx) return nullptr;
    if (opts) ctx->opts = opts->o;
    if (ctx->opts.device.empty()) ctx->opts.device = "auto";
    if (!ctx->m.load(gguf_path, ctx->opts.device, ctx->opts.n_threads)) {
        ctx->error = ctx->m.error;
    }
    return ctx;
}

void AICORE_CAPI aicore_gaussian_free(aicore_gaussian_ctx* ctx) { delete ctx; }

const char* AICORE_CAPI
aicore_gaussian_last_error(const aicore_gaussian_ctx* ctx) {
    if (!ctx) return "NULL context";
    return ctx->error.empty() ? nullptr : ctx->error.c_str();
}

// ---- model geometry ----

int AICORE_CAPI aicore_gaussian_geometry_of(const aicore_gaussian_ctx* ctx,
                                            aicore_gaussian_geometry* out) {
    if (!ctx || !out || !ctx->error.empty()) return -1;
    const aicore::gaussian::hparams& hp = ctx->m.hp();
    out->in_channels = hp.in_channels;
    out->image_height = hp.image_size;
    out->image_width = hp.image_size;
    out->gaussian_channels = hp.gaussian_channels;
    out->sh_degree = hp.sh_degree;
    return 0;
}

// ---- inference from raw float images ----

int AICORE_CAPI aicore_gaussian_run(aicore_gaussian_ctx* ctx,
                                    const float* images,
                                    int32_t n_views,
                                    int32_t height,
                                    int32_t width,
                                    float** out,
                                    size_t* n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    if (!ctx || !images || !out || !n_out) return -1;
    if (!ctx->error.empty()) return -1;

    std::vector<float> clean;
    if (!aicore::gaussian::ingest_images(ctx->m.hp(), images, n_views, height,
                                         width, clean, ctx->error)) {
        return -1;
    }

    std::vector<float> result;
    if (!ctx->m.forward(clean.data(), n_views, result)) {
        ctx->error = ctx->m.error;
        return -1;
    }

    *out = (float*)malloc(result.size() * sizeof(float));
    if (!*out) {
        ctx->error = "output allocation failed";
        return -1;
    }
    std::memcpy(*out, result.data(), result.size() * sizeof(float));
    *n_out = result.size();
    return 0;
}

void AICORE_CAPI aicore_gaussian_free_floats(float* p) { free(p); }
void AICORE_CAPI aicore_gaussian_free_bytes(unsigned char* p) { free(p); }

// ---- inference from image files ----

int AICORE_CAPI aicore_gaussian_run_paths(aicore_gaussian_ctx* ctx,
                                          const char** image_paths,
                                          int32_t n_images,
                                          float** out,
                                          size_t* n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    if (!ctx || !image_paths || n_images < 1 || !out || !n_out) return -1;
    if (!ctx->error.empty()) return -1;

    const int size = ctx->m.hp().image_size;
    std::vector<std::string> paths(n_images);
    for (int i = 0; i < n_images; i++) paths[i] = image_paths[i];

    std::vector<float> images;
    std::string err;
    if (!aicore::gaussian::load_images_chw(paths, size, images, err)) {
        ctx->error = err;
        return -1;
    }

    return aicore_gaussian_run(ctx, images.data(), n_images, size, size, out,
                               n_out);
}

// ---- pose recovery ----

int AICORE_CAPI aicore_gaussian_estimate_poses(const float* gaussians,
                                               int32_t n_views,
                                               int32_t height,
                                               int32_t width,
                                               int32_t gaussian_channels,
                                               float opacity_threshold,
                                               float* cam2world_out,
                                               float* focal_out) {
    if (!gaussians || !cam2world_out || n_views < 1 || height < 1 ||
        width < 1 || gaussian_channels < 16)
        return -1;
    const int P = height * width;
    std::vector<std::vector<float>> pts, ops;
    std::vector<const float*> pptr, optr;

    // De-interleave per-view
    pts.assign(n_views, {});
    ops.assign(n_views, {});
    pptr.resize(n_views);
    optr.resize(n_views);
    for (int v = 0; v < n_views; v++) {
        pts[v].resize((size_t)3 * P);
        ops[v].resize(P);
        for (int i = 0; i < P; i++) {
            const float* a =
                    &gaussians[(size_t)(v * P + i) * gaussian_channels];
            pts[v][3 * i] = a[0];
            pts[v][3 * i + 1] = a[1];
            pts[v][3 * i + 2] = a[2];
            ops[v][i] = a[15];
        }
        pptr[v] = pts[v].data();
        optr[v] = ops[v].data();
    }

    aicore::gaussian::pose::PoseResult pr =
            aicore::gaussian::pose::estimate_poses(pptr, optr, height, width,
                                                   opacity_threshold);

    for (int v = 0; v < n_views; v++)
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                cam2world_out[v * 16 + i * 4 + j] =
                        (float)pr.cam2world[v](i, j);
    if (focal_out) *focal_out = (float)pr.focal;
    return 0;
}

// ---- PLY export ----

int AICORE_CAPI aicore_gaussian_export_ply(const float* gaussians,
                                           int32_t n_views,
                                           int32_t height,
                                           int32_t width,
                                           int32_t gaussian_channels,
                                           int32_t sh_degree,
                                           float opacity_threshold,
                                           const char* out_ply) {
    if (!gaussians || !out_ply) return -1;
    std::string err;
    if (!aicore::gaussian::export_ply_sibr(gaussians, n_views, height, width,
                                           gaussian_channels, sh_degree,
                                           opacity_threshold, out_ply, err)) {
        return -1;
    }
    return 0;
}

int AICORE_CAPI aicore_gaussian_export_ply_bytes(const float* gaussians,
                                                 int32_t n_views,
                                                 int32_t height,
                                                 int32_t width,
                                                 int32_t gaussian_channels,
                                                 int32_t sh_degree,
                                                 float opacity_threshold,
                                                 unsigned char** out_bytes,
                                                 size_t* out_size) {
    if (!gaussians || !out_bytes || !out_size) return -1;
    *out_bytes = nullptr;
    *out_size = 0;
    std::vector<uint8_t> buf;
    std::string err;
    if (!aicore::gaussian::export_ply_sibr_to_buffer(
                gaussians, n_views, height, width, gaussian_channels, sh_degree,
                opacity_threshold, buf, err)) {
        return -1;
    }
    if (buf.empty()) return -1;
    auto* mem = static_cast<unsigned char*>(std::malloc(buf.size()));
    if (!mem) return -1;
    std::memcpy(mem, buf.data(), buf.size());
    *out_bytes = mem;
    *out_size = buf.size();
    return 0;
}

int AICORE_CAPI aicore_gaussian_run_and_export_ply(aicore_gaussian_ctx* ctx,
                                                   const char** image_paths,
                                                   int32_t n_images,
                                                   float opacity_threshold,
                                                   const char* out_ply) {
    if (!ctx || !image_paths || !out_ply) return -1;
    if (!ctx->error.empty()) return -1;

    float* gaussians = nullptr;
    size_t n_out = 0;
    if (aicore_gaussian_run_paths(ctx, image_paths, n_images, &gaussians,
                                  &n_out) != 0) {
        return -1;
    }

    const int gc = ctx->m.hp().gaussian_channels;
    const int H = ctx->m.hp().image_size;
    const int W = ctx->m.hp().image_size;
    const int sh_deg = ctx->m.hp().sh_degree;
    int ret = aicore_gaussian_export_ply(gaussians, n_images, H, W, gc, sh_deg,
                                         opacity_threshold, out_ply);
    aicore_gaussian_free_floats(gaussians);
    return ret;
}

// ---- backend warmup (main-thread CUDA init) ----

int AICORE_CAPI aicore_gaussian_warmup_backend(const char* device) {
    // Lightweight probe on the UI thread: register ggml backends and clear any
    // sticky CUDA errors from BEV/SIBR/COLMAP. Do NOT keep a backend alive here
    // — a persistent warmup backend plus the worker's backend was implicated in
    // ggml/CUDA teardown crashes when the worker freed the model context.
    (void)device;
    ggml_common::load_backends_once();
#if defined(GGML_USE_CUDA) && !defined(GGML_BACKEND_DL)
    cudaGetLastError();
#endif
    return 0;
}

// ---- model cache directory ----

char* AICORE_CAPI aicore_gaussian_model_cache_dir(void) {
    std::string dir = aicore::gaussian::default_model_cache_dir();
    char* result = static_cast<char*>(std::malloc(dir.size() + 1));
    if (result) {
        std::strcpy(result, dir.c_str());
    }
    return result;
}

void AICORE_CAPI aicore_gaussian_free_string(char* s) { free(s); }

// ---- model info ----

char* AICORE_CAPI aicore_gaussian_info_json(aicore_gaussian_ctx* ctx) {
    if (!ctx) return nullptr;
    const auto& hp = ctx->m.hp();
    char buf[1024];
    std::snprintf(buf, sizeof(buf),
                  "{\n"
                  "  \"architecture\": \"free-splatter\",\n"
                  "  \"n_layer\": %d,\n"
                  "  \"n_embd\": %d,\n"
                  "  \"n_head\": %d,\n"
                  "  \"head_dim\": %d,\n"
                  "  \"patch_size\": %d,\n"
                  "  \"image_size\": %d,\n"
                  "  \"in_channels\": %d,\n"
                  "  \"gaussian_channels\": %d,\n"
                  "  \"sh_degree\": %d,\n"
                  "  \"sh_residual\": %s,\n"
                  "  \"use_2dgs\": %s,\n"
                  "  \"device\": \"%s\"\n"
                  "}\n",
                  hp.n_layer, hp.n_embd, hp.n_head, hp.head_dim, hp.patch_size,
                  hp.image_size, hp.in_channels, hp.gaussian_channels,
                  hp.sh_degree, hp.sh_residual ? "true" : "false",
                  hp.use_2dgs ? "true" : "false", ctx->m.be.device.c_str());
    char* result = (char*)malloc(std::strlen(buf) + 1);
    if (result) std::strcpy(result, buf);
    return result;
}

// ---- CLI helpers (accumulate / parallax) ------------------------------------

struct aicore_gaussian_accumulator {
    aicore::gaussian::pose::Accumulator acc;
};

namespace {

void deinterleave_gaussian_views(const float* gaussians,
                                 int n_views,
                                 int P,
                                 int gc,
                                 std::vector<std::vector<float>>& pts,
                                 std::vector<std::vector<float>>& ops,
                                 std::vector<const float*>& pptr,
                                 std::vector<const float*>& optr) {
    pts.assign(n_views, {});
    ops.assign(n_views, {});
    pptr.resize(n_views);
    optr.resize(n_views);
    for (int v = 0; v < n_views; v++) {
        pts[v].resize((size_t)3 * P);
        ops[v].resize(P);
        for (int i = 0; i < P; i++) {
            const float* a = &gaussians[(size_t)(v * P + i) * gc];
            pts[v][3 * i] = a[0];
            pts[v][3 * i + 1] = a[1];
            pts[v][3 * i + 2] = a[2];
            ops[v][i] = a[15];
        }
        pptr[v] = pts[v].data();
        optr[v] = ops[v].data();
    }
}

void copy_accum_points(
        const std::vector<aicore::gaussian::pose::AccumPoint>& src,
        aicore_gaussian_point** out,
        size_t* n_out) {
    if (n_out) *n_out = 0;
    if (out) *out = nullptr;
    if (!out || !n_out || src.empty()) return;
    auto* buf = static_cast<aicore_gaussian_point*>(
            std::malloc(src.size() * sizeof(aicore_gaussian_point)));
    if (!buf) return;
    for (size_t i = 0; i < src.size(); ++i) {
        buf[i].x = src[i].x;
        buf[i].y = src[i].y;
        buf[i].z = src[i].z;
        buf[i].r = src[i].r;
        buf[i].g = src[i].g;
        buf[i].b = src[i].b;
        buf[i].opacity = src[i].opacity;
        buf[i].sx = src[i].sx;
        buf[i].sy = src[i].sy;
        buf[i].sz = src[i].sz;
        buf[i].qw = src[i].qw;
        buf[i].qx = src[i].qx;
        buf[i].qy = src[i].qy;
        buf[i].qz = src[i].qz;
        buf[i].frame = src[i].frame;
    }
    *out = buf;
    *n_out = src.size();
}

std::vector<aicore::gaussian::pose::AccumPoint> fs_points_to_accum(
        const aicore_gaussian_point* cloud, size_t n) {
    std::vector<aicore::gaussian::pose::AccumPoint> pts(n);
    for (size_t i = 0; i < n; ++i) {
        pts[i].x = cloud[i].x;
        pts[i].y = cloud[i].y;
        pts[i].z = cloud[i].z;
        pts[i].r = cloud[i].r;
        pts[i].g = cloud[i].g;
        pts[i].b = cloud[i].b;
        pts[i].opacity = cloud[i].opacity;
        pts[i].sx = cloud[i].sx;
        pts[i].sy = cloud[i].sy;
        pts[i].sz = cloud[i].sz;
        pts[i].qw = cloud[i].qw;
        pts[i].qx = cloud[i].qx;
        pts[i].qy = cloud[i].qy;
        pts[i].qz = cloud[i].qz;
        pts[i].frame = cloud[i].frame;
    }
    return pts;
}

}  // namespace

int AICORE_CAPI aicore_gaussian_pair_parallax(const float* gaussians,
                                              int32_t n_views,
                                              int32_t height,
                                              int32_t width,
                                              int32_t gc,
                                              float opacity_threshold,
                                              aicore_gaussian_parallax* out) {
    if (!gaussians || !out || n_views < 2) return -1;
    const int P = height * width;
    std::vector<std::vector<float>> pts, ops;
    std::vector<const float*> pptr, optr;
    deinterleave_gaussian_views(gaussians, n_views, P, gc, pts, ops, pptr,
                                optr);
    const aicore::gaussian::pose::Parallax px =
            aicore::gaussian::pose::pair_parallax(pptr, optr, height, width,
                                                  opacity_threshold);
    out->tri_angle_deg = px.tri_angle_deg;
    out->lateral_angle_deg = px.lateral_angle_deg;
    out->baseline_over_depth = px.baseline_over_depth;
    out->baseline = px.baseline;
    out->median_depth = px.median_depth;
    out->focal = px.focal;
    out->n_points = px.n_points;
    return 0;
}

aicore_gaussian_accumulator* AICORE_CAPI aicore_gaussian_accumulator_new(
        int height, int width, float opacity_threshold) {
    try {
        return new aicore_gaussian_accumulator{
                aicore::gaussian::pose::Accumulator(height, width,
                                                    opacity_threshold)};
    } catch (...) {
        return nullptr;
    }
}

void AICORE_CAPI
aicore_gaussian_accumulator_free(aicore_gaussian_accumulator* acc) {
    delete acc;
}

void AICORE_CAPI aicore_gaussian_accumulator_add_pair(
        aicore_gaussian_accumulator* acc, const float* gaussians, int gc) {
    if (acc && gaussians) acc->acc.add_pair(gaussians, gc);
}

int AICORE_CAPI
aicore_gaussian_accumulator_frame_count(aicore_gaussian_accumulator* acc) {
    return acc ? acc->acc.frame_count() : 0;
}

void AICORE_CAPI
aicore_gaussian_accumulator_cloud(aicore_gaussian_accumulator* acc,
                                  aicore_gaussian_point** out,
                                  size_t* n_out) {
    if (!acc) {
        if (out) *out = nullptr;
        if (n_out) *n_out = 0;
        return;
    }
    copy_accum_points(acc->acc.cloud(), out, n_out);
}

void AICORE_CAPI
aicore_gaussian_accumulator_refine(aicore_gaussian_accumulator* acc,
                                   float voxel_frac,
                                   int iters,
                                   float alpha) {
    if (acc) acc->acc.refine(voxel_frac, iters, alpha);
}

int AICORE_CAPI
aicore_gaussian_accumulator_fuse(aicore_gaussian_accumulator* acc,
                                 float voxel_frac,
                                 int fuse_k,
                                 int fuse_mode,
                                 aicore_gaussian_point** out,
                                 size_t* n_out) {
    if (!acc || !out || !n_out) return -1;
    std::vector<aicore::gaussian::pose::AccumPoint> fused;
    aicore::gaussian::pose::consensus_fuse(acc->acc.cloud(), voxel_frac, fuse_k,
                                           fused, fuse_mode);
    copy_accum_points(fused, out, n_out);
    return (*out && *n_out > 0) ? 0 : -1;
}

int AICORE_CAPI aicore_gaussian_tree_overlap(const float** pairs,
                                             int n_pairs,
                                             int gc,
                                             int height,
                                             int width,
                                             float opacity_threshold,
                                             int block,
                                             int overlap,
                                             int max_levels,
                                             float layout_spacing,
                                             int per_node_cap,
                                             aicore_gaussian_point** out,
                                             size_t* n_out,
                                             int* n_nodes_out) {
    if (!pairs || n_pairs < 1 || !out || !n_out) return -1;
    std::vector<aicore::gaussian::pose::AccumPoint> cloud =
            aicore::gaussian::pose::tree_accumulate_overlap(
                    std::vector<const float*>(pairs, pairs + n_pairs), height,
                    width, gc, opacity_threshold, block, overlap, 0.02, 300, 0,
                    nullptr, max_levels, layout_spacing, n_nodes_out,
                    per_node_cap);
    copy_accum_points(cloud, out, n_out);
    return (*out && *n_out > 0) ? 0 : -1;
}

int AICORE_CAPI aicore_gaussian_fuse_cloud(const aicore_gaussian_point* cloud,
                                           size_t n,
                                           float voxel_frac,
                                           int fuse_k,
                                           int fuse_mode,
                                           aicore_gaussian_point** out,
                                           size_t* n_out) {
    if (!cloud || n == 0 || !out || !n_out) return -1;
    std::vector<aicore::gaussian::pose::AccumPoint> fused;
    aicore::gaussian::pose::consensus_fuse(
            fs_points_to_accum(cloud, n), voxel_frac, fuse_k, fused, fuse_mode);
    copy_accum_points(fused, out, n_out);
    return (*out && *n_out > 0) ? 0 : -1;
}

double AICORE_CAPI aicore_gaussian_refine_cloud(aicore_gaussian_point* cloud,
                                                size_t n,
                                                float voxel_frac,
                                                int iters,
                                                float alpha) {
    if (!cloud || n == 0) return -1.0;
    auto pts = fs_points_to_accum(cloud, n);
    const double rms = aicore::gaussian::pose::consensus_refine(pts, voxel_frac,
                                                                iters, alpha);
    for (size_t i = 0; i < n; ++i) {
        cloud[i].x = pts[i].x;
        cloud[i].y = pts[i].y;
        cloud[i].z = pts[i].z;
        cloud[i].r = pts[i].r;
        cloud[i].g = pts[i].g;
        cloud[i].b = pts[i].b;
        cloud[i].opacity = pts[i].opacity;
        cloud[i].sx = pts[i].sx;
        cloud[i].sy = pts[i].sy;
        cloud[i].sz = pts[i].sz;
        cloud[i].qw = pts[i].qw;
        cloud[i].qx = pts[i].qx;
        cloud[i].qy = pts[i].qy;
        cloud[i].qz = pts[i].qz;
        cloud[i].frame = pts[i].frame;
    }
    return rms;
}
}  // extern "C"
