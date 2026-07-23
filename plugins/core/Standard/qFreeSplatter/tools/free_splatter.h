// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

// C-style CLI API for free_splatter-cli — thin wrapper over AICore C++ engine.
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct free_splatter_options free_splatter_options;
typedef struct free_splatter_ctx free_splatter_ctx;
typedef struct free_splatter_accumulator free_splatter_accumulator;

typedef struct {
    int32_t in_channels;
    int32_t image_height;
    int32_t image_width;
    int32_t gaussian_channels;
} free_splatter_geometry;

typedef struct {
    float x, y, z;
    float r, g, b, opacity;
    float sx, sy, sz;
    float qw, qx, qy, qz;
    int32_t frame;
} free_splatter_point;

typedef struct {
    double tri_angle_deg;
    double lateral_angle_deg;
    double baseline_over_depth;
    double baseline;
    double median_depth;
    double focal;
    int n_points;
} free_splatter_parallax;

free_splatter_options* free_splatter_options_new(void);
void free_splatter_options_set_device(free_splatter_options* opts,
                                      const char* device);
void free_splatter_options_set_dump_taps_dir(free_splatter_options* opts,
                                             const char* dir);
void free_splatter_options_free(free_splatter_options* opts);

free_splatter_ctx* free_splatter_load(const char* gguf_path,
                                      free_splatter_options* opts);
void free_splatter_free(free_splatter_ctx* ctx);
const char* free_splatter_last_error(free_splatter_ctx* ctx);

void free_splatter_geometry_of(free_splatter_ctx* ctx,
                               free_splatter_geometry* out);

int free_splatter_run(free_splatter_ctx* ctx,
                      const float* images,
                      int32_t n_views,
                      int32_t height,
                      int32_t width,
                      float** out,
                      size_t* n_out);
void free_splatter_buf_free(void* p);

int free_splatter_pair_parallax(const float* gaussians,
                                int32_t n_views,
                                int32_t height,
                                int32_t width,
                                int32_t gc,
                                float opacity_threshold,
                                free_splatter_parallax* out);

free_splatter_accumulator* free_splatter_accumulator_new(
        int height, int width, float opacity_threshold);
void free_splatter_accumulator_free(free_splatter_accumulator* acc);
void free_splatter_accumulator_add_pair(free_splatter_accumulator* acc,
                                        const float* gaussians,
                                        int gc);
int free_splatter_accumulator_frame_count(free_splatter_accumulator* acc);
void free_splatter_accumulator_cloud(free_splatter_accumulator* acc,
                                     free_splatter_point** out,
                                     size_t* n_out);
void free_splatter_accumulator_refine(free_splatter_accumulator* acc,
                                      float voxel_frac,
                                      int iters,
                                      float alpha);
int free_splatter_accumulator_fuse(free_splatter_accumulator* acc,
                                   float voxel_frac,
                                   int fuse_k,
                                   int fuse_mode,
                                   free_splatter_point** out,
                                   size_t* n_out);

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
                               int* n_nodes_out);

int free_splatter_fuse_cloud(const free_splatter_point* cloud,
                             size_t n,
                             float voxel_frac,
                             int fuse_k,
                             int fuse_mode,
                             free_splatter_point** out,
                             size_t* n_out);

double free_splatter_refine_cloud(free_splatter_point* cloud,
                                  size_t n,
                                  float voxel_frac,
                                  int iters,
                                  float alpha);

int free_splatter_export_splat(const float* gaussians,
                               size_t count,
                               int gaussian_channels,
                               float opacity_threshold,
                               size_t max_splats,
                               const char* output_path);
int free_splatter_export_cloud_splat(const free_splatter_point* cloud,
                                     size_t count,
                                     size_t max_splats,
                                     float scale_multiplier,
                                     const char* output_path);

#ifdef __cplusplus
}
#endif
