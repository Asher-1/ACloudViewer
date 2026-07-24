// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// Thin CLI adapter — forwards to exported aicore_gaussian_* symbols in
// libAICore.so.
#include <new>
#include <vector>

#include "aicore/gaussian_capi.h"
#include "free_splatter.h"

struct free_splatter_options {
    aicore_gaussian_options* inner;
};

struct free_splatter_ctx {
    aicore_gaussian_ctx* inner;
};

struct free_splatter_accumulator {
    aicore_gaussian_accumulator* inner;
};

extern "C" {

free_splatter_options* free_splatter_options_new(void) {
    auto* opts = new (std::nothrow) free_splatter_options();
    if (!opts) return nullptr;
    opts->inner = aicore_gaussian_options_new();
    if (!opts->inner) {
        delete opts;
        return nullptr;
    }
    return opts;
}

void free_splatter_options_set_device(free_splatter_options* opts,
                                      const char* device) {
    if (opts && opts->inner)
        aicore_gaussian_options_set_device(opts->inner, device);
}

void free_splatter_options_set_dump_taps_dir(free_splatter_options* opts,
                                             const char* dir) {
    if (opts && opts->inner)
        aicore_gaussian_options_set_dump_taps_dir(opts->inner, dir);
}

void free_splatter_options_free(free_splatter_options* opts) {
    if (!opts) return;
    aicore_gaussian_options_free(opts->inner);
    delete opts;
}

free_splatter_ctx* free_splatter_load(const char* gguf_path,
                                      free_splatter_options* opts) {
    if (!gguf_path) return nullptr;
    auto* ctx = new (std::nothrow) free_splatter_ctx();
    if (!ctx) return nullptr;
    if (opts && opts->inner) {
        ctx->inner = aicore_gaussian_load_opts(gguf_path, opts->inner);
    } else {
        aicore_gaussian_options* o = aicore_gaussian_options_new();
        if (o) {
            aicore_gaussian_options_set_device(o, "auto");
            ctx->inner = aicore_gaussian_load_opts(gguf_path, o);
            aicore_gaussian_options_free(o);
        }
    }
    if (!ctx->inner) {
        delete ctx;
        return nullptr;
    }
    return ctx;
}

void free_splatter_free(free_splatter_ctx* ctx) {
    if (!ctx) return;
    aicore_gaussian_free(ctx->inner);
    delete ctx;
}

const char* free_splatter_last_error(free_splatter_ctx* ctx) {
    if (!ctx || !ctx->inner) return "NULL context";
    return aicore_gaussian_last_error(ctx->inner);
}

void free_splatter_geometry_of(free_splatter_ctx* ctx,
                               free_splatter_geometry* out) {
    if (!ctx || !ctx->inner || !out) return;
    aicore_gaussian_geometry geo{};
    if (aicore_gaussian_geometry_of(ctx->inner, &geo) != 0) return;
    out->in_channels = geo.in_channels;
    out->image_height = geo.image_height;
    out->image_width = geo.image_width;
    out->gaussian_channels = geo.gaussian_channels;
}

int free_splatter_run(free_splatter_ctx* ctx,
                      const float* images,
                      int32_t n_views,
                      int32_t height,
                      int32_t width,
                      float** out,
                      size_t* n_out) {
    if (!ctx || !ctx->inner) return -1;
    return aicore_gaussian_run(ctx->inner, images, n_views, height, width, out,
                               n_out);
}

void free_splatter_buf_free(void* p) {
    aicore_gaussian_free_floats(static_cast<float*>(p));
}

int free_splatter_pair_parallax(const float* gaussians,
                                int32_t n_views,
                                int32_t height,
                                int32_t width,
                                int32_t gc,
                                float opacity_threshold,
                                free_splatter_parallax* out) {
    if (!out) return -1;
    aicore_gaussian_parallax px{};
    if (aicore_gaussian_pair_parallax(gaussians, n_views, height, width, gc,
                                      opacity_threshold, &px) != 0) {
        return -1;
    }
    out->tri_angle_deg = px.tri_angle_deg;
    out->lateral_angle_deg = px.lateral_angle_deg;
    out->baseline_over_depth = px.baseline_over_depth;
    out->baseline = px.baseline;
    out->median_depth = px.median_depth;
    out->focal = px.focal;
    out->n_points = px.n_points;
    return 0;
}

free_splatter_accumulator* free_splatter_accumulator_new(
        int height, int width, float opacity_threshold) {
    auto* acc = new (std::nothrow) free_splatter_accumulator();
    if (!acc) return nullptr;
    acc->inner =
            aicore_gaussian_accumulator_new(height, width, opacity_threshold);
    if (!acc->inner) {
        delete acc;
        return nullptr;
    }
    return acc;
}

void free_splatter_accumulator_free(free_splatter_accumulator* acc) {
    if (!acc) return;
    aicore_gaussian_accumulator_free(acc->inner);
    delete acc;
}

void free_splatter_accumulator_add_pair(free_splatter_accumulator* acc,
                                        const float* gaussians,
                                        int gc) {
    if (acc && acc->inner)
        aicore_gaussian_accumulator_add_pair(acc->inner, gaussians, gc);
}

int free_splatter_accumulator_frame_count(free_splatter_accumulator* acc) {
    return acc && acc->inner
                   ? aicore_gaussian_accumulator_frame_count(acc->inner)
                   : 0;
}

void free_splatter_accumulator_cloud(free_splatter_accumulator* acc,
                                     free_splatter_point** out,
                                     size_t* n_out) {
    if (out) *out = nullptr;
    if (n_out) *n_out = 0;
    if (!acc || !acc->inner || !out || !n_out) return;
    aicore_gaussian_point* pts = nullptr;
    aicore_gaussian_accumulator_cloud(acc->inner, &pts, n_out);
    *out = reinterpret_cast<free_splatter_point*>(pts);
}

void free_splatter_accumulator_refine(free_splatter_accumulator* acc,
                                      float voxel_frac,
                                      int iters,
                                      float alpha) {
    if (acc && acc->inner)
        aicore_gaussian_accumulator_refine(acc->inner, voxel_frac, iters,
                                           alpha);
}

int free_splatter_accumulator_fuse(free_splatter_accumulator* acc,
                                   float voxel_frac,
                                   int fuse_k,
                                   int fuse_mode,
                                   free_splatter_point** out,
                                   size_t* n_out) {
    if (!acc || !acc->inner) return -1;
    aicore_gaussian_point* pts = nullptr;
    const int rc = aicore_gaussian_accumulator_fuse(
            acc->inner, voxel_frac, fuse_k, fuse_mode, &pts, n_out);
    *out = reinterpret_cast<free_splatter_point*>(pts);
    return rc;
}

int free_splatter_tree_overlap(const float** pairs,
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
                               free_splatter_point** out,
                               size_t* n_out,
                               int* n_nodes_out) {
    aicore_gaussian_point* pts = nullptr;
    const int rc = aicore_gaussian_tree_overlap(
            pairs, n_pairs, gc, height, width, opacity_threshold, block,
            overlap, max_levels, layout_spacing, per_node_cap, &pts, n_out,
            n_nodes_out);
    *out = reinterpret_cast<free_splatter_point*>(pts);
    return rc;
}

int free_splatter_fuse_cloud(const free_splatter_point* cloud,
                             size_t n,
                             float voxel_frac,
                             int fuse_k,
                             int fuse_mode,
                             free_splatter_point** out,
                             size_t* n_out) {
    aicore_gaussian_point* pts = nullptr;
    const int rc = aicore_gaussian_fuse_cloud(
            reinterpret_cast<const aicore_gaussian_point*>(cloud), n,
            voxel_frac, fuse_k, fuse_mode, &pts, n_out);
    *out = reinterpret_cast<free_splatter_point*>(pts);
    return rc;
}

double free_splatter_refine_cloud(free_splatter_point* cloud,
                                  size_t n,
                                  float voxel_frac,
                                  int iters,
                                  float alpha) {
    return aicore_gaussian_refine_cloud(
            reinterpret_cast<aicore_gaussian_point*>(cloud), n, voxel_frac,
            iters, alpha);
}

int free_splatter_export_splat(const float* gaussians,
                               size_t count,
                               int gaussian_channels,
                               float opacity_threshold,
                               size_t max_splats,
                               const char* output_path) {
    return aicore_gaussian_export_splat(gaussians, count, gaussian_channels,
                                        opacity_threshold, max_splats,
                                        output_path);
}

int free_splatter_export_cloud_splat(const free_splatter_point* cloud,
                                     size_t count,
                                     size_t max_splats,
                                     float scale_multiplier,
                                     const char* output_path) {
    if (!cloud) return -1;
    std::vector<aicore_gaussian_point> points(count);
    for (size_t i = 0; i < count; ++i) {
        points[i] = {cloud[i].x,  cloud[i].y,  cloud[i].z,       cloud[i].r,
                     cloud[i].g,  cloud[i].b,  cloud[i].opacity, cloud[i].sx,
                     cloud[i].sy, cloud[i].sz, cloud[i].qw,      cloud[i].qx,
                     cloud[i].qy, cloud[i].qz, cloud[i].frame};
    }
    return aicore_gaussian_export_cloud_splat(points.data(), points.size(),
                                              max_splats, scale_multiplier,
                                              output_path);
}

}  // extern "C"
