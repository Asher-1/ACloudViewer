// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <stddef.h>
#include <stdint.h>

#include "aicore/export.h"

#ifndef AICORE_GAUSSIAN_CAPI_H
#define AICORE_GAUSSIAN_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/* ABI version. */
AICORE_CAPI int aicore_gaussian_abi_version(void);

typedef struct aicore_gaussian_ctx aicore_gaussian_ctx;

/* ---- options builder ---- */
typedef struct aicore_gaussian_options aicore_gaussian_options;
AICORE_CAPI aicore_gaussian_options* aicore_gaussian_options_new(void);
AICORE_CAPI void aicore_gaussian_options_free(aicore_gaussian_options* opts);
/* device: NULL or "cpu", "gpu", "cuda", "vulkan" (optionally ":N"). */
AICORE_CAPI void aicore_gaussian_options_set_device(
        aicore_gaussian_options* opts, const char* device);
/* n_threads <= 0 picks a default (CPU only). */
AICORE_CAPI void aicore_gaussian_options_set_threads(
        aicore_gaussian_options* opts, int n_threads);
AICORE_CAPI void aicore_gaussian_options_set_dump_taps_dir(
        aicore_gaussian_options* opts, const char* dir);

/* ---- lifecycle ---- */
/* Load a GGUF model. Returns NULL on failure (see aicore_gaussian_last_error).
 */
AICORE_CAPI aicore_gaussian_ctx* aicore_gaussian_load(const char* gguf_path,
                                                      int n_threads);
AICORE_CAPI aicore_gaussian_ctx* aicore_gaussian_load_opts(
        const char* gguf_path, const aicore_gaussian_options* opts);
AICORE_CAPI void aicore_gaussian_free(aicore_gaussian_ctx* ctx);
AICORE_CAPI const char* aicore_gaussian_last_error(
        const aicore_gaussian_ctx* ctx);

/* ---- model geometry ---- */
typedef struct {
    int32_t in_channels;
    int32_t image_height;
    int32_t image_width;
    int32_t gaussian_channels;
    int32_t sh_degree;
} aicore_gaussian_geometry;
AICORE_CAPI int aicore_gaussian_geometry_of(const aicore_gaussian_ctx* ctx,
                                            aicore_gaussian_geometry* out);

/* ---- inference from raw float images ---- */
/* images: n_views * in_channels * height * width float32, range [0,1], NCHW.
   On success *out is malloc'd: n_views * height * width * gaussian_channels
   float32. Free with aicore_gaussian_free_floats. Returns 0 on success, -1 on
   failure. */
AICORE_CAPI int aicore_gaussian_run(aicore_gaussian_ctx* ctx,
                                    const float* images,
                                    int32_t n_views,
                                    int32_t height,
                                    int32_t width,
                                    float** out,
                                    size_t* n_out);
AICORE_CAPI void aicore_gaussian_free_floats(float* p);

/* ---- inference from image files ---- */
/* Load N image files, preprocess (center-crop, resize to model resolution),
   run inference. image_paths has n_images entries. On success *out is malloc'd.
 */
AICORE_CAPI int aicore_gaussian_run_paths(aicore_gaussian_ctx* ctx,
                                          const char** image_paths,
                                          int32_t n_images,
                                          float** out,
                                          size_t* n_out);
AICORE_CAPI void aicore_gaussian_free_bytes(unsigned char* p);

/* ---- pose recovery ---- */
/* Recover each view's camera from engine output. cam2world_out: n_views*16
 * float32. */
AICORE_CAPI int aicore_gaussian_estimate_poses(const float* gaussians,
                                               int32_t n_views,
                                               int32_t height,
                                               int32_t width,
                                               int32_t gaussian_channels,
                                               float opacity_threshold,
                                               float* cam2world_out,
                                               float* focal_out);

/* ---- PLY export (SIBR-compatible) ---- */
/* Export engine output as a PLY file for SIBR Gaussian viewer.
   gaussians: n*height*width*gaussian_channels float32.
   sh_degree: SH degree of the model (0-3). opacity_threshold: prune threshold.
   Returns 0 on success, -1 on failure. */
AICORE_CAPI int aicore_gaussian_export_ply(const float* gaussians,
                                           int32_t n_views,
                                           int32_t height,
                                           int32_t width,
                                           int32_t gaussian_channels,
                                           int32_t sh_degree,
                                           float opacity_threshold,
                                           const char* out_ply);

/* Export SIBR-compatible binary PLY into memory. Caller frees with
 * aicore_gaussian_free_bytes. */
AICORE_CAPI int aicore_gaussian_export_ply_bytes(const float* gaussians,
                                                 int32_t n_views,
                                                 int32_t height,
                                                 int32_t width,
                                                 int32_t gaussian_channels,
                                                 int32_t sh_degree,
                                                 float opacity_threshold,
                                                 unsigned char** out_bytes,
                                                 size_t* out_size);

/* Convenience: run inference from image files and export PLY in one call. */
AICORE_CAPI int aicore_gaussian_run_and_export_ply(aicore_gaussian_ctx* ctx,
                                                   const char** image_paths,
                                                   int32_t n_images,
                                                   float opacity_threshold,
                                                   const char* out_ply);

/* Initialize ggml backends on the calling thread (CUDA-safe when invoked from
 * the UI thread before worker inference). Returns 0 on success, -1 on failure.
 */
AICORE_CAPI int aicore_gaussian_warmup_backend(const char* device);

/* ---- model cache directory ---- */
/* Default cross-platform GGUF model cache directory (UTF-8). Free with
 * aicore_gaussian_free_string. */
AICORE_CAPI char* aicore_gaussian_model_cache_dir(void);
AICORE_CAPI void aicore_gaussian_free_string(char* s);

/* ---- model info ---- */
/* malloc'd JSON describing model config; free via aicore_gaussian_free_string.
 */
AICORE_CAPI char* aicore_gaussian_info_json(aicore_gaussian_ctx* ctx);

/* ---- CLI helpers (accumulate / parallax / splat export pipeline) ---- */
typedef struct {
    double tri_angle_deg;
    double lateral_angle_deg;
    double baseline_over_depth;
    double baseline;
    double median_depth;
    double focal;
    int n_points;
} aicore_gaussian_parallax;

typedef struct {
    float x, y, z;
    float r, g, b, opacity;
    float sx, sy, sz;
    float qw, qx, qy, qz;
    int32_t frame;
} aicore_gaussian_point;

typedef struct aicore_gaussian_accumulator aicore_gaussian_accumulator;

/* Export activated model output to an antimatter15 .splat file. `count` is
 * the number of gaussian records, not the number of floats. max_splats=0 keeps
 * all records that pass opacity_threshold. */
AICORE_CAPI int aicore_gaussian_export_splat(const float* gaussians,
                                             size_t count,
                                             int32_t gaussian_channels,
                                             float opacity_threshold,
                                             size_t max_splats,
                                             const char* output_path);

/* Export an accumulated/fused cloud to .splat. scale_multiplier=1 preserves
 * the predicted anisotropic radii; max_splats=0 keeps all points. */
AICORE_CAPI int aicore_gaussian_export_cloud_splat(
        const aicore_gaussian_point* cloud,
        size_t count,
        size_t max_splats,
        float scale_multiplier,
        const char* output_path);

AICORE_CAPI int aicore_gaussian_pair_parallax(const float* gaussians,
                                              int32_t n_views,
                                              int32_t height,
                                              int32_t width,
                                              int32_t gc,
                                              float opacity_threshold,
                                              aicore_gaussian_parallax* out);

AICORE_CAPI aicore_gaussian_accumulator* aicore_gaussian_accumulator_new(
        int height, int width, float opacity_threshold);
AICORE_CAPI void aicore_gaussian_accumulator_free(
        aicore_gaussian_accumulator* acc);
AICORE_CAPI void aicore_gaussian_accumulator_add_pair(
        aicore_gaussian_accumulator* acc, const float* gaussians, int gc);
AICORE_CAPI int aicore_gaussian_accumulator_frame_count(
        aicore_gaussian_accumulator* acc);
AICORE_CAPI void aicore_gaussian_accumulator_cloud(
        aicore_gaussian_accumulator* acc,
        aicore_gaussian_point** out,
        size_t* n_out);
AICORE_CAPI void aicore_gaussian_accumulator_refine(
        aicore_gaussian_accumulator* acc,
        float voxel_frac,
        int iters,
        float alpha);
AICORE_CAPI int aicore_gaussian_accumulator_fuse(
        aicore_gaussian_accumulator* acc,
        float voxel_frac,
        int fuse_k,
        int fuse_mode,
        aicore_gaussian_point** out,
        size_t* n_out);

AICORE_CAPI int aicore_gaussian_tree_overlap(const float** pairs,
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
                                             int* n_nodes_out);

AICORE_CAPI int aicore_gaussian_fuse_cloud(const aicore_gaussian_point* cloud,
                                           size_t n,
                                           float voxel_frac,
                                           int fuse_k,
                                           int fuse_mode,
                                           aicore_gaussian_point** out,
                                           size_t* n_out);

AICORE_CAPI double aicore_gaussian_refine_cloud(aicore_gaussian_point* cloud,
                                                size_t n,
                                                float voxel_frac,
                                                int iters,
                                                float alpha);

#ifdef __cplusplus
}
#endif

#endif  // AICORE_GAUSSIAN_CAPI_H
