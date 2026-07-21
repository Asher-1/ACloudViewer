// ----------------------------------------------------------------------------
// -                        CloudViewer: www.cloudViewer.org                  -
// ----------------------------------------------------------------------------
// Copyright (c) 2018-2024 www.cloudViewer.org
// SPDX-License-Identifier: MIT
// ----------------------------------------------------------------------------

#pragma once
#include <stddef.h>

#include "aicore/export.h"

#ifndef AICORE_DEPTH_CAPI_H
#define AICORE_DEPTH_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif
typedef struct aicore_depth_ctx aicore_depth_ctx;
/* ABI version. 3: added aicore_depth_depth_dense, aicore_depth_points,
   aicore_depth_free_bytes. 4: added aicore_depth_load_nested (two-branch metric
   model).
 */
int AICORE_CAPI aicore_depth_abi_version(void);
aicore_depth_ctx* AICORE_CAPI
aicore_depth_load(const char* gguf_path, int n_threads); /* NULL on failure */
/* Load a NESTED metric model from its two branches: the anyview (GIANT) GGUF
   and the metric (ViT-L + DPT/sky) GGUF. The returned ctx runs the nested
   metric alignment: aicore_depth_depth_dense / aicore_depth_depth_path /
   aicore_depth_pose_path all produce the final metric-scale depth + scaled
   extrinsics (is_metric=1, conf/sky = NULL). NULL on failure. */
aicore_depth_ctx* aicore_depth_load_nested(const char* anyview_gguf,
                                           const char* metric_gguf,
                                           int n_threads);
void aicore_depth_free(aicore_depth_ctx* ctx); /* safe on NULL */
/* malloc'd JSON describing model config; free via aicore_depth_free_string. */
char* aicore_depth_info_json(aicore_depth_ctx* ctx);
void aicore_depth_free_string(char* s);
const char* aicore_depth_last_error(
        aicore_depth_ctx* ctx); /* owned by ctx, "" if none */
/* Run depth on an image file. On success writes *out_h,*out_w and returns a
   malloc'd float[H*W] depth map (row-major); caller frees via
   aicore_depth_free_floats. NULL on error. */
float* aicore_depth_depth_path(aicore_depth_ctx* ctx,
                               const char* image_path,
                               int* out_h,
                               int* out_w);
void aicore_depth_free_floats(float* p);
/* Run pose; fills ext[12] (3x4 row-major) and intr[9] (3x3). Returns 0 ok, -1
 * error. */
int aicore_depth_pose_path(aicore_depth_ctx* ctx,
                           const char* image_path,
                           float out_ext[12],
                           float out_intr[9]);
/* Multi-view depth+pose. n_images paths. Fills, per view i: out_ext[i*12],
   out_intr[i*9]. Returns a malloc'd float[n*H*W] depth (view-major), sets
   *out_h,*out_w,*out_n; NULL on error. Caller frees the returned buffer via
   aicore_depth_free_floats. */
float* aicore_depth_depth_pose_multi(aicore_depth_ctx* ctx,
                                     const char** image_paths,
                                     int n_images,
                                     int* out_h,
                                     int* out_w,
                                     int* out_n,
                                     float* out_ext /* n*12 */,
                                     float* out_intr /* n*9 */);
/* Single-image 3D export. Runs the native depth+pose pipeline, captures the
   processed-resolution RGB colors, and writes a glTF-2.0 binary point cloud to
   out_glb. Returns 0 ok, -1 error (see aicore_depth_last_error). */
int aicore_depth_export_glb(aicore_depth_ctx* ctx,
                            const char* image_path,
                            const char* out_glb);
/* Single-image 3D export to a COLMAP sparse model (cameras/images/points3D) in
   directory out_dir. binary != 0 => .bin (default); 0 => .txt. Returns 0 ok, -1
   error. */
int aicore_depth_export_colmap(aicore_depth_ctx* ctx,
                               const char* image_path,
                               const char* out_dir,
                               int binary);
/* Multi-view COLMAP sparse export: multiview depth+pose, back-project to
   points3D. image_paths has n_images entries; writes under out_dir. Returns 0
   ok, -1 error. */
int aicore_depth_export_colmap_multi(aicore_depth_ctx* ctx,
                                     const char** image_paths,
                                     int n_images,
                                     const char* out_dir,
                                     int binary);
/* Same as aicore_depth_export_colmap_multi but image_names[i] is COLMAP
   Image.Name() (relative to the image root). NULL names fall back to
   basename(image_paths[i]). */
int aicore_depth_export_colmap_multi_named(aicore_depth_ctx* ctx,
                                           const char** image_paths,
                                           const char** image_names,
                                           int n_images,
                                           const char* out_dir,
                                           int binary);
/* Write COLMAP sparse from an existing multiview depth+pose result (no
   re-inference). depth is n*h*w row-major; ext is n*12 row-major 3x4; intr is
   n*9 row-major 3x3. */
int aicore_depth_write_colmap_from_multiview(aicore_depth_ctx* ctx,
                                             const char** image_paths,
                                             const char** image_names,
                                             int n_images,
                                             const float* depth,
                                             const float* ext,
                                             const float* intr,
                                             int h,
                                             int w,
                                             const char* out_dir,
                                             int binary);

/* Dense per-pixel output for a single image. Returns 0 ok, -1 error.
   Writes processed dims to *out_h,*out_w. Each non-NULL out_* float buffer is
   malloc'd [H*W] row-major and must be freed via aicore_depth_free_floats;
   buffers not produced by the model are set to NULL.
     - DualDPT model (camera-pose capable): *out_depth + *out_conf are filled,
       *out_sky = NULL, out_ext[12] (3x4 row-major) + out_intr[9] (3x3) filled.
     - mono model (DA3MONO): *out_depth + *out_sky are filled, *out_conf = NULL,
       out_ext/out_intr zeroed (mono has no camera pose).
     - nested model (aicore_depth_load_nested): *out_depth = final metric-scale
   depth, *out_conf = *out_sky = NULL, out_ext/out_intr = scaled
   extrinsics/intrinsics. *out_is_metric = 1 for metric/nested/mono variants
   (best-effort from config), else 0. Any of
   out_h/out_w/out_depth/out_conf/out_sky/out_is_metric may be NULL;
   out_ext/out_intr must point to 12/9 floats respectively. */
int aicore_depth_depth_dense(aicore_depth_ctx* ctx,
                             const char* image_path,
                             int* out_h,
                             int* out_w,
                             float** out_depth,
                             float** out_conf,
                             float** out_sky,
                             float out_ext[12],
                             float out_intr[9],
                             int* out_is_metric);

/* Single-image 3D point cloud (DualDPT/pose-capable models only; returns -1 for
   mono models with a clear last_error). Runs depth+pose+processed-RGB,
   back-projects to world space keeping pixels with conf >= conf_thresh. On
   success sets *out_n and writes a malloc'd *out_xyz[3*N float] + *out_rgb[3*N
   uint8]; free xyz via aicore_depth_free_floats and rgb via
   aicore_depth_free_bytes. Returns 0 ok, -1 error. */
int aicore_depth_points(aicore_depth_ctx* ctx,
                        const char* image_path,
                        float conf_thresh,
                        int* out_n,
                        float** out_xyz,
                        unsigned char** out_rgb);
/* Free a uint8 buffer returned by aicore_depth_points (out_rgb). */
void aicore_depth_free_bytes(unsigned char* p);
/* Default cross-platform GGUF model cache directory (UTF-8). Free with
 * aicore_depth_free_string. */
char* AICORE_CAPI aicore_depth_model_cache_dir(void);
/* Override preprocess longest-side target before inference (0 = model default).
 * Clamped to >= patch_size. Safe to call on NULL ctx (no-op). */
void AICORE_CAPI aicore_depth_set_img_resize_target(aicore_depth_ctx* ctx,
                                                    int target);
/* Drop ggml graph buffers and (when GPU offloading) device-resident weights.
 * Call between sequential per-view inferences to keep VRAM peak O(1). */
void AICORE_CAPI aicore_depth_release_gpu_working_memory(aicore_depth_ctx* ctx);
/* Cap preprocess long-edge from free GPU VRAM (single-view activation peak).
 * Returns min(requested, vram-safe cap). No-op on CPU / when ctx is NULL. */
int AICORE_CAPI aicore_depth_cap_img_resize_target(aicore_depth_ctx* ctx,
                                                   int requested);
/* Lightweight main-thread backend warmup: register ggml backends and clear
 * sticky CUDA errors. Returns 0 on success. */
int AICORE_CAPI aicore_depth_warmup_backend(const char* device);
#ifdef __cplusplus
}
#endif

#endif  // AICORE_DEPTH_CAPI_H
